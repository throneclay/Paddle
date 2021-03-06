/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*
 * Data flow graph is an pass that build the basic graph. It contains a graph
 * and the iterators that enable the iteration over the graph.
 */

#pragma once

#include <deque>
#include <stack>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/inference/analysis/graph_traits.h"
#include "paddle/fluid/inference/analysis/node.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * DataFlowGraph - A container of Value and Function Nodes.
 */
struct DataFlowGraph {
  NodeMap nodes;
  std::vector<Node *> inputs;
  std::vector<Node *> outputs;

  // Extract inputs and outputs of the graph.
  void Build();

  // Output a DOT graph file for debug.
  std::string DotString() const;

 private:
  // Remove duplicate edges and so on.
  void Clean();
};

/*
 * An graph trait help to traverse the graph using BFS.
 * The BFS start from a graph's inputs, the graph should be fully-connected, so
 * that the iterator can reach the end.
 */
template <>
struct GraphTraits<DataFlowGraph> {
  // BFS iterator on nodes.
  struct NodesBFSIterator
      : public std::iterator<std::forward_iterator_tag, Node *> {
    NodesBFSIterator() = default;
    explicit NodesBFSIterator(const std::vector<Node *> &source);
    // NodesBFSIterator(NodesBFSIterator &&other) noexcept;
    // NOTE Heavy to use.
    NodesBFSIterator(const NodesBFSIterator &other);

    Node &operator*();
    NodesBFSIterator &operator++();
    Node *operator->();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesBFSIterator &operator=(const NodesBFSIterator &other);
    bool operator==(const NodesBFSIterator &other);
    bool operator!=(const NodesBFSIterator &other) { return !(*this == other); }

   private:
    std::deque<Node *> queue_;
    std::unordered_set<Node *> visited_;
  };

  // DFS iterator on nodes.
  struct NodesDFSIterator
      : public std::iterator<std::forward_iterator_tag, Node *> {
    NodesDFSIterator() = default;
    explicit NodesDFSIterator(const std::vector<Node *> &source);
    // NodesDFSIterator(NodesDFSIterator &&other) noexcept;
    NodesDFSIterator(const NodesDFSIterator &other);

    Node &operator*();
    NodesDFSIterator &operator++();
    // TODO(Superjomn) current implementation just compare the first
    // element, need to compare the graph and all the elements in the queue and
    // set.
    NodesDFSIterator &operator=(const NodesDFSIterator &other);
    bool operator==(const NodesDFSIterator &other);
    bool operator!=(const NodesDFSIterator &other) { return !(*this == other); }
    Node *operator->();

   private:
    std::stack<Node *> stack_;
    std::unordered_set<Node *> visited_;
  };

  explicit GraphTraits(DataFlowGraph *graph) : graph_(graph) {}

  // default use BFS to visit the nodes.
  iterator_range<NodesBFSIterator> nodes() {
    return iterator_range<NodesBFSIterator>(nodes_bfs_begin(), nodes_bfs_end());
  }
  iterator_range<NodesBFSIterator> nodes_in_BFS() {
    return iterator_range<NodesBFSIterator>(nodes_bfs_begin(), nodes_bfs_end());
  }
  iterator_range<NodesDFSIterator> nodes_in_DFS() {
    return iterator_range<NodesDFSIterator>(nodes_dfs_begin(), nodes_dfs_end());
  }

 private:
  NodesBFSIterator nodes_bfs_begin() {
    return NodesBFSIterator(graph_->inputs);
  }
  NodesBFSIterator nodes_bfs_end() { return NodesBFSIterator(); }
  NodesDFSIterator nodes_dfs_begin() {
    return NodesDFSIterator(graph_->inputs);
  }
  NodesDFSIterator nodes_dfs_end() { return NodesDFSIterator(); }

 private:
  DataFlowGraph *graph_;
};

// Extract the inputs and outputs of a graph. The inputs and outputs of a
// sub-graph is the inputs nodes and output nodes that doesn't inside the
// sub-graph.
static std::pair<std::vector<Node *>, std::vector<Node *>>
ExtractInputAndOutputOfSubGraph(std::vector<Node *> &graph) {  // NOLINT
  std::unordered_set<Node *> nodes(graph.begin(), graph.end());
  std::unordered_set<Node *> inputs;
  std::unordered_set<Node *> outputs;
  // Input a Value, check whether its inlink is in the subgraph.
  auto inlink_in_subgraph = [&](Node *n) {
    for (auto *in : n->inlinks) {
      if (nodes.count(in)) return true;
    }
    return false;
  };
  for (auto &node : graph) {
    for (auto *in : node->inlinks) {
      // The Value that is written by nodes inside a sub-graph shouldn't be the
      // input of the sub-graph.
      if (!nodes.count(in) && in->type() == Node::Type::kValue &&
          !inlink_in_subgraph(in)) {
        inputs.insert(in);
      }
    }
    for (auto *out : node->outlinks) {
      if (!nodes.count(out) && out->type() == Node::Type::kValue) {
        outputs.insert(out);
      }
    }
  }
  return std::make_pair(std::vector<Node *>(inputs.begin(), inputs.end()),
                        std::vector<Node *>(outputs.begin(), outputs.end()));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
