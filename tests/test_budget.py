"""CPU-only coverage-order tests using actual, uneven timestamps."""

from itertools import permutations
from types import SimpleNamespace
import unittest

from analyzer.budget import coverage_order


def frames(times):
    return [SimpleNamespace(timestamp_sec=t) for t in times]


class CoverageOrderTests(unittest.TestCase):
    def test_first_three_and_five_cover_time_not_index_positions(self):
        pool = frames([0, .1, .2, 24, 49, 50, 75, 99, 100])
        order = coverage_order(pool, list(range(len(pool))))
        self.assertEqual(order[:3], [0, 8, 5])
        self.assertEqual([pool[i].timestamp_sec for i in order[:5]], [0, 100, 50, 24, 75])
        self.assertEqual(sorted(order), list(range(len(pool))))
        early = sorted(pool[i].timestamp_sec for i in order[:5])
        self.assertEqual(max(b-a for a, b in zip(early, early[1:])), 26)

    def test_gap_and_midpoint_ties_prefer_earlier_time(self):
        self.assertEqual(coverage_order(frames([0, 2, 4, 6, 8]), [0, 1, 2, 3, 4]), [0, 4, 2, 1, 3])
        self.assertEqual(coverage_order(frames([0, 4, 6, 10]), [0, 1, 2, 3]), [0, 3, 1, 2])

    def test_only_selected_frames_are_returned_even_when_pool_has_other_endpoints(self):
        pool = frames([float(i) for i in range(14)])
        self.assertEqual(coverage_order(pool, [9, 3, 7, 4]), [3, 9, 7, 4])

    def test_empty_one_and_two_selected_frames(self):
        self.assertEqual(coverage_order([], []), [])
        pool = frames([0.0333, .5333, 1.0333])
        self.assertEqual(coverage_order(pool, []), [])
        self.assertEqual(coverage_order(pool, [1]), [1])
        self.assertEqual(coverage_order(tuple(pool), (2, 0)), [0, 2])

    def test_invalid_or_non_increasing_pts_rejected_even_outside_selection(self):
        for times in ([0, float('nan'), 2], [0, float('inf'), 2], [-1, 0, 2],
                      [0, True, 2], [0, '1', 2], [0, None, 2], [0, 0, 2], [0, 2, 1],
                      [0, 10**1000, 10**1001]):
            with self.subTest(times=times):
                with self.assertRaises(ValueError):
                    coverage_order(frames(times), [0])
        with self.assertRaises(ValueError):
            coverage_order([object()], [])

    def test_invalid_indices_are_not_silently_deduplicated_coerced_or_clipped(self):
        pool = frames([0, 1, 2])
        for indices in ([0, 0], [-1], [3], [False], [1.0], ['1'], [None]):
            with self.subTest(indices=indices):
                with self.assertRaises(ValueError):
                    coverage_order(pool, indices)
        for pool_arg, indices_arg in ((None, []), ({}, []), (pool, None), (pool, {0, 1})):
            with self.subTest(pool=pool_arg, indices=indices_arg):
                with self.assertRaises(TypeError):
                    coverage_order(pool_arg, indices_arg)

    def test_input_objects_are_not_mutated_and_image_bytes_are_never_accessed(self):
        class Frame:
            def __init__(self, timestamp):
                self.timestamp_sec = timestamp

            @property
            def jpeg(self):
                raise AssertionError('Coverage ordering must not access images')

        pool = [Frame(t) for t in (0.0333, .5333, 1.1333, 2.0333)]
        selected = [3, 1, 0]
        original_objects = tuple(pool)
        original_times = [f.timestamp_sec for f in pool]
        result = coverage_order(pool, selected)
        self.assertEqual(result, [0, 3, 1])
        self.assertEqual(selected, [3, 1, 0])
        self.assertEqual(tuple(pool), original_objects)
        self.assertEqual([f.timestamp_sec for f in pool], original_times)
        self.assertIsNot(result, selected)

    def test_input_permutations_have_identical_full_pass_and_coverage_order(self):
        pool = frames([0.0333, .5333, 1.0333, 5.0333, 5.5333, 10.0333])
        selected = [0, 1, 3, 5]
        expected = coverage_order(pool, selected)
        for permutation in permutations(selected):
            result = coverage_order(pool, permutation)
            self.assertEqual(result, expected)
            self.assertEqual(sorted(result), selected)
            self.assertEqual(len(set(result)), len(selected))


if __name__ == '__main__':
    unittest.main()
