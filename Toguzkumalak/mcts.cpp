#include "mcts.h"
#include <math.h>

bool Node::isFullyExpanded()
{
    return children.size() == action_idxes.size();
}

Node* Node::bestChild(float explorationWeight)
{
    Node* bestChild = nullptr;
    float bestValue = std::numeric_limits<float>::lowest();
    for (auto child : children)
    {
        float value = (child->value / (child->visits + 1e-6)) + explorationWeight * std::sqrt(std::log(visits + 1.0f) / (child->visits + 1e-6));
        if (value > bestValue)
        {
            bestValue = value;
            bestChild = child;
        }
    }
    return bestChild;
}

void Node::expand(Node* node)
{
    action_idxes = game->getPossibleMoves();
}



