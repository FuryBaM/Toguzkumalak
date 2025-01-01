from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool
from game_cython cimport Game

cdef extern from "node.h":
    cdef cppclass UCTNode:
        Game* game
        size_t action_size
        int move
        bool is_expanded
        bool self_play
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

        UCTNode(Game* game, int move, UCTNode* parent, bool self_play)
        void DestroyAllChildren()
        UCTNode* select_leaf()
        void backup(float value_estimate)
        void expand(vector[float] child_priors)
        int best_child()
        UCTNode* try_add_child(int move)