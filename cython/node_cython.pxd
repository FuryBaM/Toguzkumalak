from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from game_cython cimport Game

cdef extern from "node.h":
    vector[float] generate_dirichlet_noise(size_t size)
    void clearTree(UCTNode* node)
    cdef cppclass UCTNode:
        Game* game
        size_t action_size
        int move
        bool is_expanded
        bool self_play
        bool is_root
        UCTNode* parent
        unordered_map[int, UCTNode*] children
        vector[float] child_priors
        vector[float] child_total_value
        vector[float] child_number_visits
        vector[int] action_idxes
        float a

        float getNumberVisits()
        void setNumberVisits(float value)
        float getTotalValue()
        void setTotalValue(float value)

        UCTNode(Game* game, int move, UCTNode* parent, bool self_play, bool is_root)
        void destroyChildren()
        void destroyAllChildren()
        UCTNode* select_leaf()
        void backup(float value_estimate)
        void expand(vector[float] child_priors)
        int best_child()
        UCTNode* try_add_child(int move)
        vector[float] add_dirichlet_noise(vector[float] action_idxs, vector[float] child_priors)