#ifndef _ADJ_LIST_H_
#define _ADJ_LIST_H_

#include <set>
#include <unordered_map>

class AdjList
{
private:
    int vertices;
    int edges;
    std::unordered_map<int, std::set<int>> data;
public:
    AdjList(const char* path);
    ~AdjList();

    int num_vertices() const;
    int num_edges() const;
};

#endif