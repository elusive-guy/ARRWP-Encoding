
import numpy as np

from utils import read_test_walks, generate_walks, seed_everything

from graphgps.transform.approx_rw_utils import count_walks
from gt_funcs import gt_count_walks

from graphgps.transform.approx_rw_transforms\
    import (calculate_arrwp_matrix,
            calculate_arwse_matrix,
            calculate_arwpe_matrix)
from gt_funcs import (gt_calculate_arrwp_matrix,
                      gt_calculate_arwse_matrix,
                      gt_calculate_arwpe_matrix)


class TestWalksCounter:
    def test1(self):
        walks, num_nodes = read_test_walks()

        window_size = 10

        vec = count_walks(walks, num_nodes, window_size)
        gt_vec = gt_count_walks(walks, num_nodes, window_size)
        
        assert (vec == gt_vec).all()

    def test2(self):
        walks, num_nodes = read_test_walks()

        walk_length = walks.shape[1]
        window_size = walk_length

        vec = count_walks(walks, num_nodes, window_size)
        gt_vec = gt_count_walks(walks, num_nodes, window_size)
        
        assert (vec == gt_vec).all()

    def test3(self):
        walks_ = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ])

        vec = count_walks(walks_, 5, 3, replace_zeros=False)
        gt_vec = np.array([1, 2, 1, 0, 0])

        assert (vec == gt_vec).all()

    def test4(self):
        walks_ = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ])

        vec = count_walks(walks_, 5, 3, replace_zeros=True)
        gt_vec = np.array([1, 2, 1, 1, 1])

        assert (vec == gt_vec).all()


class TestARRWPMatrix:
    def test_small(self):
        walks, num_nodes = read_test_walks()
        
        walk_length = walks.shape[1]
        window_size_lst = [10, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                mx = calculate_arrwp_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )
                gt_mx = gt_calculate_arrwp_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(mx.to_dense(), gt_mx)

    def test_medium(self):
        seed_everything(42)
        
        num_nodes = 100
        num_walks = 100
        walk_length = 60

        many_walks = generate_walks(
            num_nodes, num_walks, walk_length,
        )

        window_size_lst = [40, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                mx = calculate_arrwp_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )
                gt_mx = gt_calculate_arrwp_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(mx.to_dense(), gt_mx, rtol=1e-4)


class TestARWPEMatrix:
    def test_small(self):
        walks, num_nodes = read_test_walks()
        
        walk_length = walks.shape[1]
        window_size_lst = [10, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                mx = calculate_arwpe_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )
                gt_mx = gt_calculate_arwpe_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(mx, gt_mx)

    def test_medium(self):
        seed_everything(42)

        num_nodes = 100
        num_walks = 100
        walk_length = 60

        many_walks = generate_walks(
            num_nodes, num_walks, walk_length,
        )

        window_size_lst = [40, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                mx = calculate_arwpe_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )
                gt_mx = gt_calculate_arwpe_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(mx, gt_mx)


class TestARWSEMatrix:
    def test_small(self):
        walks, num_nodes = read_test_walks()
        
        walk_length = walks.shape[1]
        window_size_lst = [10, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                mx = calculate_arwse_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )
                gt_mx = gt_calculate_arwse_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(mx, gt_mx)

    def test_medium(self):
        seed_everything(42)

        num_nodes = 100
        num_walks = 100
        walk_length = 60

        many_walks = generate_walks(
            num_nodes, num_walks, walk_length,
        )

        window_size_lst = [40, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                mx = calculate_arwse_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )
                gt_mx = gt_calculate_arwse_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(mx, gt_mx)