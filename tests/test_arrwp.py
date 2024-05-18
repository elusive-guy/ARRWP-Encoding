import torch
import numpy as np

from utils import (read_test_walks, generate_walks,
                   get_edge_index, seed_everything)

from gt_funcs import gt_count_walks
from graphgps.transform.approx_rw_utils import (remove_extra_loops,
                                                check_loops,
                                                count_walks,
                                                get_relations_for_stats,
                                                get_relations,
                                                add_self_relations_for_stats,
                                                add_self_relations)

from gt_funcs import (gt_calculate_arrwp_matrix,
                      gt_calculate_arwse_matrix,
                      gt_calculate_arwpe_matrix)
from graphgps.transform.approx_rw_transforms\
    import (calculate_arrwpe_stats,
            calculate_arrwp_matrix_for_stats,
            calculate_arrwp_matrix,
            calculate_arwse_matrix,
            calculate_arwpe_matrix)


class TestWalksCounter:
    def test1(self):
        walks, num_nodes = read_test_walks()

        window_size = 10

        gt_vec = gt_count_walks(walks, num_nodes, window_size)
        vec = count_walks(walks, num_nodes, window_size)
        
        assert (gt_vec == vec).all()

    def test2(self):
        walks, num_nodes = read_test_walks()

        walk_length = walks.shape[1]
        window_size = walk_length

        gt_vec = gt_count_walks(walks, num_nodes, window_size)
        vec = count_walks(walks, num_nodes, window_size)
        
        assert (gt_vec == vec).all()

    def test3(self):
        walks = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ])

        gt_vec = np.array([1, 2, 1, 0, 0])
        vec = count_walks(walks, 5, 3, replace_zeros=False)

        assert (gt_vec == vec).all()

    def test4(self):
        walks = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ])

        gt_vec = np.array([1, 2, 1, 1, 1])
        vec = count_walks(walks, 5, 3, replace_zeros=True)

        assert (gt_vec == vec).all()


class TestLoopsFuncs:
    def test_remove_extra_loops_1(self):
        edge_index = torch.Tensor([
            [2, 3, 1, 0, 3, 1, 1, 4, 3, 4, 0],
            [2, 4, 1, 4, 3, 0, 3, 4, 2, 3, 0],
        ]).to(torch.long)
        self.check_remove_extra_loops_test_(edge_index)

    def test_remove_extra_loops_2(self):
        edge_index = torch.Tensor([
            [3, 0, 3, 1, 1, 4, 3, 4, 0],
            [4, 4, 3, 0, 3, 4, 2, 3, 0],
        ]).to(torch.long)
        self.check_remove_extra_loops_test_(edge_index)

    def test_remove_extra_loops_3(self):
        edge_index = torch.Tensor([
            [3, 0, 1, 1, 3, 4],
            [4, 4, 0, 3, 2, 3],
        ]).to(torch.long)
        self.check_remove_extra_loops_test_(edge_index)

    def test_check_loops_1(self):
        edge_index = torch.Tensor([
            [0, 0, 3, 1, 1, 2, 3, 4],
            [1, 0, 2, 1, 4, 2, 3, 4],
        ])
        num_nodes = 5
        assert check_loops(edge_index, num_nodes) == True
    
    def test_check_loops_2(self):
        edge_index = torch.Tensor([
            [0, 0, 3, 1, 4, 1, 2, 3],
            [1, 0, 2, 1, 4, 4, 2, 3],
        ])
        num_nodes = 5
        assert check_loops(edge_index, num_nodes) == False

    def test_check_loops_3(self):
        edge_index = torch.Tensor([
            [0, 0, 3, 1, 1, 3, 4],
            [1, 0, 2, 1, 4, 3, 4],
        ])
        num_nodes = 5
        assert check_loops(edge_index, num_nodes) == False

    def test_check_loops_4(self):
        edge_index = torch.Tensor([
            [0, 3, 1],
            [1, 2, 4],
        ])
        num_nodes = 5
        assert check_loops(edge_index, num_nodes) == False

    def check_remove_extra_loops_test_(self, edge_index):
        sur_index = torch.Tensor([
            [0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4],
            [0, 4, 0, 1, 3, 2, 2, 3, 4, 3, 4],
        ]).to(torch.long)
        sur_val = torch.rand((sur_index.shape[1], 7))

        upd_sur_index, upd_sur_val = remove_extra_loops(
            sur_index, sur_val, edge_index,
        )

        gt_st = set(tuple(edge.tolist()) for edge in edge_index.T)
        st = set(tuple(edge.tolist()) for edge in upd_sur_index.T)

        assert gt_st == st

        gt_vals = torch.sparse_coo_tensor(
            sur_index, sur_val,
        )

        for idx in range(upd_sur_index.shape[1]):
            src = upd_sur_index[0, idx]
            dst = upd_sur_index[1, idx]

            gt_val = gt_vals[src, dst]
            val = upd_sur_val[idx]

            assert torch.allclose(gt_val, val)


class TestRelationsFuncs:
    def test_get_relations_for_stats_1(self):
        walks = np.array([
            [0, 1, 2, 3, 4],
            [2, 1, 0, 0, 4],
            [4, 4, 4, 4, 3],
        ])
        window_size = 2

        gt_relations = np.array([
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            [0, 1, 4, 1, 0, 2, 1, 2, 3, 3, 4, 3, 4],
        ])
        relations = get_relations_for_stats(walks, window_size)

        gt_st = set(tuple(rel) for rel in gt_relations.T)
        st = set(tuple(rel) for rel in relations.T)

        assert gt_relations.shape == relations.shape\
                        and gt_st == st

    def test_get_relations_for_stats_2(self):
        walks = np.array([
            [0, 1, 2, 3, 4],
            [2, 1, 0, 0, 4],
            [4, 4, 4, 4, 3],
        ])
        window_size = 3

        gt_relations = np.array([
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4],
            [0, 1, 2, 4, 0, 1, 2, 3, 0, 1, 2, 3, 4, 3, 4],
        ])
        relations = get_relations_for_stats(walks, window_size)

        gt_st = set(tuple(rel) for rel in gt_relations.T)
        st = set(tuple(rel) for rel in relations.T)

        assert gt_relations.shape == relations.shape\
                        and gt_st == st
    
    def test_get_relations_for_stats_3(self):
        walks = np.array([
            [0, 1, 2, 3, 4],
            [2, 1, 0, 0, 4],
            [4, 4, 4, 4, 3],
        ])
        window_size = 5

        gt_relations = np.array([
            [0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4],
            [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4],
        ])
        relations = get_relations_for_stats(walks, window_size)

        gt_st = set(tuple(rel) for rel in gt_relations.T)
        st = set(tuple(rel) for rel in relations.T)

        assert gt_relations.shape == relations.shape\
                        and gt_st == st
        
    def test_get_relations_1(self):
        walks = np.array([
            [0, 1, 2, 3, 4],
            [2, 1, 0, 0, 4],
            [4, 4, 4, 4, 3],
        ])
        window_size = 2

        gt_relations = np.array([
            [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4],
            [0, 0, 1, 4, 1, 0, 2, 1, 2, 3, 3, 4, 3, 4, 4],
            [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        ])
        relations = get_relations(walks, window_size)

        gt_st = set(tuple(rel) for rel in gt_relations.T)
        st = set(tuple(rel) for rel in relations.T)

        assert gt_relations.shape == relations.shape\
                        and gt_st == st

    def test_get_relations_2(self):
        walks = np.array([
            [0, 1, 2, 3, 4],
            [2, 1, 0, 0, 4],
            [4, 4, 4, 4, 3],
        ])
        window_size = 3

        gt_relations = np.array([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
             2, 2, 2, 2, 2, 4, 4, 4, 4],
            [0, 0, 1, 2, 4, 0, 0, 1, 2, 3,
             0, 1, 2, 3, 4, 3, 4, 4, 4],
            [0, 1, 1, 2, 2, 1, 2, 0, 1, 2,
             2, 1, 0, 1, 2, 2, 0, 1, 2],
        ])
        relations = get_relations(walks, window_size)

        gt_st = set(tuple(rel) for rel in gt_relations.T)
        st = set(tuple(rel) for rel in relations.T)

        assert gt_relations.shape == relations.shape\
                        and gt_st == st
    
    def test_get_relations_3(self):
        walks = np.array([
            [0, 1, 2, 3, 4],
            [2, 1, 0, 0, 4],
            [4, 4, 4, 4, 3],
        ])
        window_size = 5

        gt_relations = np.array([
            [0, 0, 0, 0, 0, 2, 2, 2,
             2, 2, 4, 4, 4, 4, 4],
            [0, 1, 2, 3, 4, 0, 0, 1,
             2, 4, 3, 4, 4, 4, 4],
            [0, 1, 2, 3, 4, 2, 3, 1,
             0, 4, 4, 0, 1, 2, 3],
        ])
        relations = get_relations(walks, window_size)

        gt_st = set(tuple(rel) for rel in gt_relations.T)
        st = set(tuple(rel) for rel in relations.T)

        assert gt_relations.shape == relations.shape\
                        and gt_st == st
    
    def test_add_self_relations_for_stats(self):
        relations = np.array([
            [0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4],
            [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4],
        ])
        num_nodes = 5

        gt_new_relations = np.array([
            [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 4, 4],
            [0, 1, 2, 3, 4, 1, 0, 1, 2, 4, 3, 3, 4],
        ])
        new_relations = add_self_relations_for_stats(relations, num_nodes)

        gt_st = set(tuple(rel) for rel in gt_new_relations.T)
        st = set(tuple(rel) for rel in new_relations.T)

        assert gt_new_relations.shape == new_relations.shape\
                            and gt_st == st
        
    def test_add_self_relations(self):
        relations = np.array([
            [0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4],
            [0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 4],
        ])
        num_nodes = 5

        window_size = 4

        gt_new_relations_ = np.array([
            [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 4, 4],
            [0, 1, 2, 3, 4, 1, 0, 1, 2, 4, 3, 3, 4],
        ])
        num_un_relations = gt_new_relations_.shape[1]
        gt_new_relations = np.empty(
            (3, num_un_relations*window_size),
        )
        for i in range(num_un_relations):
            for depth in range(window_size):
                gt_new_relations.T[num_un_relations*depth + i] =\
                    tuple((*gt_new_relations_.T[i], depth))

        new_relations = add_self_relations(
            relations, num_nodes, window_size,
        )

        gt_st = set(tuple(rel) for rel in gt_new_relations.T)
        st = set(tuple(rel) for rel in new_relations.T)

        assert gt_new_relations.shape == new_relations.shape\
                            and gt_st == st


class TestARRWPEncoding:
    def test_arrwp_small(self):
        walks, num_nodes = read_test_walks()
        
        walk_length = walks.shape[1]
        window_size_lst = [10, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                gt_mx = gt_calculate_arrwp_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )
                mx = calculate_arrwp_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(gt_mx, mx.to_dense())

    def test_arrwp_medium(self):
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
                gt_mx = gt_calculate_arrwp_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )
                mx = calculate_arrwp_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(gt_mx, mx.to_dense(), rtol=1e-4)

    def test_arrwp_for_stats_small(self):
        walks, num_nodes = read_test_walks()
        
        walk_length = walks.shape[1]
        window_size_lst = [10, walk_length]
        scales = [False, True]

        num_edges = int(num_nodes*num_nodes*0.8)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        for window_size in window_size_lst:
            for scale in scales:
                gt_mx = gt_calculate_arrwp_matrix(
                    walks, num_nodes, window_size, scale=scale,
                    edge_index=edge_index, self_loops=True,
                )
                mx = calculate_arrwp_matrix_for_stats(
                    walks, num_nodes, window_size, scale=scale,
                    edge_index=edge_index.cpu().detach().numpy(),
                )

                assert np.allclose(gt_mx, mx.to_dense())

    def test_arrwp_for_stats_medium(self):
        seed_everything(42)
        
        num_nodes = 100
        num_walks = 100
        walk_length = 60

        many_walks = generate_walks(
            num_nodes, num_walks, walk_length,
        )

        window_size_lst = [40, walk_length]
        scales = [False, True]

        num_edges = int(num_nodes*num_nodes*0.8)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        for window_size in window_size_lst:
            for scale in scales:
                gt_mx = gt_calculate_arrwp_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                    edge_index=edge_index, self_loops=True,
                )
                mx = calculate_arrwp_matrix_for_stats(
                    many_walks, num_nodes, window_size, scale=scale,
                    edge_index=edge_index.cpu().detach().numpy(),
                )

                assert np.allclose(gt_mx, mx.to_dense(), rtol=1e-4)

    def test_arrwpe_tiny(self):
        walks = np.array([
            [0, 1, 1, 3],
            [0, 0, 2, 4],
            [4, 1, 3, 3],
            [3, 1, 4, 2],
        ])
        num_nodes = 6
        window_size = 3

        edge_index = torch.Tensor([
            [0, 1, 1, 0, 0, 2, 4, 3, 3, 1, 4],
            [1, 1, 3, 0, 2, 4, 1, 3, 1, 4, 2],
        ]).to(torch.long)

        window_size_lst = [2, 3, 4]
        scales = [False, True]

        for window_size in window_size_lst:
            size = (num_nodes, num_nodes, window_size)
            for scale in scales:
                gt_arwse = gt_calculate_arwse_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )
                gt_arrwp = gt_calculate_arrwp_matrix(
                    walks, num_nodes, window_size,
                    scale=scale, edge_index=edge_index,
                )

                abs_enc, rel_enc_idx, rel_enc_val = calculate_arrwpe_stats(
                    walks, num_nodes, window_size,
                    scale=scale, edge_index=edge_index,
                )
                rel_enc = torch.sparse_coo_tensor(
                    rel_enc_idx, rel_enc_val, size,
                )

                assert np.allclose(gt_arwse, abs_enc)
                assert np.allclose(gt_arrwp, rel_enc.to_dense())

    def test_arrwpe_small(self):
        walks, num_nodes = read_test_walks()
        edge_index = get_edge_index(walks)

        walk_length = walks.shape[1]
        window_size_lst = [10, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            size = (num_nodes, num_nodes, window_size)
            for scale in scales:
                gt_arwse = gt_calculate_arwse_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )
                gt_arrwp = gt_calculate_arrwp_matrix(
                    walks, num_nodes, window_size,
                    scale=scale, edge_index=edge_index,
                )

                abs_enc, rel_enc_idx, rel_enc_val = calculate_arrwpe_stats(
                    walks, num_nodes, window_size,
                    scale=scale, edge_index=edge_index,
                )
                rel_enc = torch.sparse_coo_tensor(
                    rel_enc_idx, rel_enc_val, size,
                )

                assert np.allclose(gt_arwse, abs_enc)
                assert np.allclose(gt_arrwp, rel_enc.to_dense())

    def test_arrwpe_medium(self):
        seed_everything(42)
        
        num_nodes = 100
        num_walks = 100
        walk_length = 60

        many_walks = generate_walks(
            num_nodes, num_walks, walk_length,
        )
        edge_index = get_edge_index(many_walks)

        window_size_lst = [40, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            size = (num_nodes, num_nodes, window_size)
            for scale in scales:
                gt_arwse = gt_calculate_arwse_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )
                gt_arrwp = gt_calculate_arrwp_matrix(
                    many_walks, num_nodes, window_size,
                    scale=scale, edge_index=edge_index,
                )

                abs_enc, rel_enc_idx, rel_enc_val = calculate_arrwpe_stats(
                    many_walks, num_nodes, window_size,
                    scale=scale, edge_index=edge_index,
                )
                rel_enc = torch.sparse_coo_tensor(
                    rel_enc_idx, rel_enc_val, size,
                )

                assert np.allclose(gt_arwse, abs_enc)
                assert np.allclose(gt_arrwp, rel_enc.to_dense())


class TestARWPEMatrix:
    def test_small(self):
        walks, num_nodes = read_test_walks()
        
        walk_length = walks.shape[1]
        window_size_lst = [10, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                gt_mx = gt_calculate_arwpe_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )
                mx = calculate_arwpe_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(gt_mx, mx)

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
                gt_mx = gt_calculate_arwpe_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )
                mx = calculate_arwpe_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(gt_mx, mx)


class TestARWSEMatrix:
    def test_small(self):
        walks, num_nodes = read_test_walks()
        
        walk_length = walks.shape[1]
        window_size_lst = [10, walk_length]
        scales = [False, True]

        for window_size in window_size_lst:
            for scale in scales:
                gt_mx = gt_calculate_arwse_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )
                mx = calculate_arwse_matrix(
                    walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(gt_mx, mx)

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
                gt_mx = gt_calculate_arwse_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )
                mx = calculate_arwse_matrix(
                    many_walks, num_nodes, window_size, scale=scale,
                )

                assert np.allclose(gt_mx, mx)