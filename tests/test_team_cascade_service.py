"""Optional cascade service path: synthetic JPEGs and CPU fake clients only."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import Mock

import pytest

from analyzer import service
from analyzer.outerwear import outerwear_policy_fingerprint
from analyzer.outerwear_cascade import CascadeOuterwearObserver, cascade_policy_fingerprint
from analyzer.video import VideoFrame
from backend.validation import validate_result
from tests.test_outerwear import primary, inventory
from tests.test_outerwear_cascade import RuntimeClient
from tests.test_rule_service import image_bytes, synthetic_report, Clock

RULE = 'removed_outerwear_in_cash_workplace'


@pytest.fixture
def setup(tmp_path, monkeypatch):
    rules = deepcopy(service.RULES)
    rules[RULE]['enabled'] = True
    monkeypatch.setattr(service, 'RULES', rules)
    monkeypatch.setattr(service, 'check_ready', Mock(return_value=[]))
    monkeypatch.setattr(service, 'LocalModelClient', Mock(side_effect=AssertionError('No real client in cascade tests')))
    frames = [VideoFrame(float(i*4), image_bytes(i), 80, 60) for i in range(3)]
    monkeypatch.setattr('analyzer.outerwear_pass.sample_video', Mock(return_value={'frames': frames}))
    source = tmp_path/'synthetic.mov'
    source.write_bytes(b'CPU fixture; decoder and all model transports are replaced')
    state = {'source': source, 'out': tmp_path/'job', 'frames': frames, 'cash_mode': 'positive', 'cash_mutate': lambda r: None}
    def cash(source, directory, **kwargs):
        report = synthetic_report(Path(directory), state['cash_mode'])
        report['settings'] = {'enable_actions': False, 'max_motion_windows': 0, 'max_refinement_frames': 0}
        for row in report['observations']:
            row['calls'] = [{'completed': True, 'finish_reason': 'stop', 'valid_json': True, 'request_seconds': 0.01}]
        report['request_count'] = len(report['observations'])
        state['cash_mutate'](report)
        return report
    state['cash_mock'] = Mock(side_effect=cash)
    monkeypatch.setattr(service, 'observe_video', state['cash_mock'])
    return state


def observer(first=('no','uncertain','uncertain'), second=('yes','yes')):
    a = RuntimeClient(4, [primary(state) for state in first])
    responses = []
    for state in second:
        responses.append(primary(state))
        if state == 'yes':
            responses.append(inventory())
    b = RuntimeClient(8, responses)
    return CascadeOuterwearObserver(a, b, expected_runtimes={'filter4b': a.runtime, 'review8b': b.runtime})


def run(state, candidate):
    return service.analyze_to_directory(state['source'], state['out'], outerwear_observer=candidate)


def saved(state):
    return json.loads((state['out']/'observations/report.json').read_text(encoding='utf-8'))


def test_supported_combined_export_has_original_source_bytes_and_separate_policy(setup):
    candidate = observer()
    snapshot = deepcopy(candidate.expected_runtimes)
    result = run(setup, candidate)
    assert validate_result(result) == result
    assert [r['timecode_sec'] for r in result['violations']] == [2.4667, 4.0]
    assert [len(r['evidence_frames']) for r in result['violations']] == [3, 2]
    for exported, frame in zip(result['violations'][1]['evidence_frames'], setup['frames'][1:]):
        assert base64.b64decode(exported['image_base64']) == frame.jpeg
        assert exported['timestamp_sec'] == frame.timestamp_sec
    metadata = json.loads((setup['out']/'job.json').read_text())
    report = saved(setup)
    assert metadata['outerwear_expected_runtimes'] == snapshot == report['outerwear_expected_runtimes']
    assert metadata['outerwear_policy_sha256'] == report['outerwear_policy_sha256'] == cascade_policy_fingerprint()
    assert report['outerwear_policy_sha256'] != outerwear_policy_fingerprint()
    assert all(stage['record']['policy_sha256'] == outerwear_policy_fingerprint()
               for row in report['outerwear_checks'] for stage in row['stages'])
    assert [len(row['calls']) for row in report['outerwear_checks']] == [1, 3, 3]
    service.LocalModelClient.assert_not_called()


def test_complete_negative_is_scoped_and_no_filter_rows_dispatch_review(setup):
    setup['cash_mode'] = 'negative'
    candidate = observer(('no','no','no'), ())
    assert run(setup, candidate) == {'verdict': 'no_violation', 'violations': []}
    assert candidate.clients['review8b'].info_calls == 0
    assert candidate.clients['review8b'].calls == []


def test_explicit_cascade_requires_enabled_rule_and_exact_class_before_models(setup, monkeypatch):
    candidate = observer()
    service.RULES[RULE]['enabled'] = False
    with pytest.raises(ValueError):
        run(setup, candidate)
    service.RULES[RULE]['enabled'] = True
    with pytest.raises(ValueError):
        run(setup, object())
    class Other(CascadeOuterwearObserver):
        pass
    other = Other(candidate.clients['filter4b'], candidate.clients['review8b'], expected_runtimes=candidate.expected_runtimes)
    with pytest.raises(ValueError):
        run(setup, other)
    service.check_ready.assert_not_called()
    setup['cash_mock'].assert_not_called()
    assert not setup['out'].exists()


@pytest.mark.parametrize('mutation', [
    lambda r: r.pop('outerwear_expected_runtimes'),
    lambda r: r['outerwear_expected_runtimes']['review8b'].update(build_info='changed'),
    lambda r: r.update(outerwear_observer_kind='original'),
    lambda r: r.update(outerwear_policy_sha256=outerwear_policy_fingerprint()),
    lambda r: r['outerwear_checks'][0].update(state='yes'),
    lambda r: r['outerwear_checks'][1]['calls'].pop(),
    lambda r: r['outerwear_checks'][0].update(frame_id='other'),
    lambda r: r['outerwear_checks'][0].update(timestamp_sec=1.0),
    lambda r: r['outerwear_checks'][0]['stages'][0]['runtime'].update(build_info='other'),
    lambda r: r['outerwear_checks'][0].update(extra='untracked'),
    lambda r: r['selected_outerwear_frame_ids'].reverse(),
    lambda r: r['outerwear_checks'].pop(),
])
def test_unverified_cascade_never_reaches_evaluation_or_export(setup, monkeypatch, mutation):
    import analyzer.outerwear_pass as module
    original = module.observe_outerwear_video
    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        mutation(report)
        return report
    monkeypatch.setattr(module, 'observe_outerwear_video', changed)
    evaluate = Mock(wraps=service.evaluate_outerwear)
    monkeypatch.setattr(service, 'evaluate_outerwear', evaluate)
    with pytest.raises(service.IncompleteAnalysisError) as caught:
        run(setup, observer())
    assert caught.value.inference_completed is False
    evaluate.assert_not_called()
    assert not (setup['out']/'result.json').exists()
    diagnostic = json.loads((setup['out']/'rule_evaluation.json').read_text())
    assert diagnostic['cascade_validated'] is False
    assert diagnostic['checks']['cash_visible_after_person_disappears']['candidates']


def test_source_jpeg_tamper_on_negative_non_evidence_row_also_blocks_export(setup, monkeypatch):
    import analyzer.outerwear_pass as module
    original = module.observe_outerwear_video
    def changed(source, directory, **kwargs):
        report = original(source, directory, **kwargs)
        (directory/report['outerwear_checks'][0]['image_path']).write_bytes(image_bytes(7))
        return report
    monkeypatch.setattr(module, 'observe_outerwear_video', changed)
    with pytest.raises(service.IncompleteAnalysisError):
        run(setup, observer())
    assert not (setup['out']/'result.json').exists()


def test_offline_export_requires_independent_runtime_snapshot_and_revalidates(setup):
    candidate = observer()
    run(setup, candidate)
    report, directory = saved(setup), setup['out']/'observations'
    with pytest.raises(service.IncompleteAnalysisError):
        service.evaluate_report(report)
    with pytest.raises(service.IncompleteAnalysisError):
        service.result_from_report(report, directory)
    assert service.result_from_report(report, directory, outerwear_expected_runtimes=candidate.expected_runtimes)['verdict'] == 'violation'
    report['outerwear_checks'][1]['stages'][1]['record']['state'] = 'no'
    with pytest.raises(service.IncompleteAnalysisError):
        service.result_from_report(report, directory, outerwear_expected_runtimes=candidate.expected_runtimes)


def test_mutated_observer_snapshot_cannot_replace_initial_job_expectations(setup, monkeypatch):
    import analyzer.outerwear_pass as module
    original = module.observe_outerwear_video
    candidate = observer()
    def changed(*args, **kwargs):
        candidate.expected_runtimes['review8b']['build_info'] = 'changed'
        candidate.clients['review8b'].runtime['build_info'] = 'changed'
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'observe_outerwear_video', changed)
    with pytest.raises(service.IncompleteAnalysisError):
        run(setup, candidate)
    assert not (setup['out']/'result.json').exists()


def test_export_revalidates_bytes_again_after_successful_evaluation(setup, monkeypatch):
    original = service._save
    changed = []
    def save(path, value, **kwargs):
        original(path, value, **kwargs)
        if path.name == 'rule_evaluation.json' and value.get('kind') == 'combined_visual_candidates_not_legal_verdict':
            report = saved(setup)
            path = setup['out']/'observations'/report['outerwear_checks'][0]['image_path']
            path.write_bytes(image_bytes(7))
            changed.append(path)
    monkeypatch.setattr(service, '_save', save)
    with pytest.raises(service.IncompleteAnalysisError) as caught:
        run(setup, observer())
    assert caught.value.inference_completed is False and len(changed) == 1
    assert not (setup['out']/'result.json').exists()


def test_source_revalidation_time_is_inside_the_same_job_budget(setup, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(service.time, 'perf_counter', clock)
    original = service._frame
    def slow_frame(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now += 21
        return result
    monkeypatch.setattr(service, '_frame', slow_frame)
    with pytest.raises(TimeoutError):
        service.analyze_to_directory(setup['source'], setup['out'], budget_sec=20, outerwear_observer=observer())
    assert not (setup['out']/'result.json').exists()


@pytest.mark.parametrize('failure', [primary('wrong'), TimeoutError('synthetic deadline')])
def test_partial_or_failed_cascade_never_uses_original_refusal_proof(setup, failure):
    candidate = observer(('uncertain',), ())
    candidate.clients['review8b'].responses = [failure]
    with pytest.raises(service.IncompleteAnalysisError) as caught:
        run(setup, candidate)
    assert caught.value.inference_completed is False
    report = saved(setup)
    assert service._refusal_inference_completed(report, {}) is False
    assert not (setup['out']/'result.json').exists()
    assert len(candidate.clients['filter4b'].calls) == 1
    assert len(candidate.clients['review8b'].calls) == 1


def test_complete_uncertain_cascade_refusal_remains_unsafe(setup):
    setup['cash_mode'] = 'negative'
    candidate = observer(('uncertain','uncertain','uncertain'), ('uncertain','uncertain','uncertain'))
    with pytest.raises(service.IncompleteAnalysisError) as caught:
        run(setup, candidate)
    assert caught.value.inference_completed is False
    assert saved(setup)['outerwear_status'] == 'completed'
    assert not (setup['out']/'result.json').exists()


def test_cash_technical_guard_still_prevents_all_cascade_calls(setup):
    setup['cash_mutate'] = lambda r: r.update(status='partial', all_selected_frames_processed=False)
    candidate = observer()
    with pytest.raises(service.IncompleteAnalysisError):
        run(setup, candidate)
    assert all(not client.calls and client.info_calls == 0 for client in candidate.clients.values())
    assert not (setup['out']/'observations/outerwear-report.json').exists()


def test_shared_deadline_before_optional_pass_prevents_dispatch(setup, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(service.time, 'perf_counter', clock)
    setup['cash_mutate'] = lambda r: setattr(clock, 'now', clock.now + 18)
    candidate = observer()
    with pytest.raises(TimeoutError):
        service.analyze_to_directory(setup['source'], setup['out'], budget_sec=20, outerwear_observer=candidate)
    assert all(not client.calls for client in candidate.clients.values())
    assert not (setup['out']/'result.json').exists()


def test_default_enabled_pass_kwargs_remain_unchanged(setup, monkeypatch):
    import analyzer.outerwear_pass as module
    from tests.test_multi_rule_service import add_outerwear
    client = object()
    monkeypatch.setattr(service, 'LocalModelClient', Mock(return_value=client))
    def original(source, directory, **kwargs):
        assert set(kwargs) == {'client', 'deadline'}
        assert kwargs['client'] is client
        return add_outerwear({}, directory, ('no','no','no'))
    monkeypatch.setattr(module, 'observe_outerwear_video', original)
    assert service.analyze_to_directory(setup['source'], setup['out'])['verdict'] == 'violation'
    metadata = json.loads((setup['out']/'job.json').read_text())
    assert 'outerwear_expected_runtimes' not in metadata


def test_default_cash_only_path_and_analyze_video_signature_unchanged(setup, monkeypatch):
    import analyzer.outerwear_pass as module
    service.RULES[RULE]['enabled'] = False
    optional = Mock(side_effect=AssertionError('Default cash cannot call optional pass'))
    monkeypatch.setattr(module, 'observe_outerwear_video', optional)
    assert service.analyze_to_directory(setup['source'], setup['out'])['verdict'] == 'violation'
    optional.assert_not_called()
    metadata = json.loads((setup['out']/'job.json').read_text())
    assert metadata['rule_scope'] == ['cash_visible_after_person_disappears']
    assert 'outerwear_observer_kind' not in metadata
    import inspect
    assert 'outerwear_observer' not in inspect.signature(service.analyze_video).parameters
