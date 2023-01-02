from ._internal import (build_matrix_with_index_value, is_linear_independent, is_space_equivalent,
        get_hs_orthogonal_basis, reduce_matrix_space, get_hermite_channel_matrix_space,
        matrix_space_to_kraus_op, kraus_op_to_matrix_space, find_closed_vector_in_space)

from ._hierarchy import (get_antisymmetric_basis, get_symmetric_basis, project_to_antisymmetric_basis,
            tensor2d_project_to_antisym_basis, tensor2d_project_to_sym_antisym_basis, has_rank,
            permutation_with_antisymmetric_factor)
# for unittest only
from ._hierarchy import naive_antisym_sym_projector, naive_tensor2d_project_to_sym_antisym_basis

from ._gradient import DetectMatrixSpaceRank, DetectExtendibleChannel
