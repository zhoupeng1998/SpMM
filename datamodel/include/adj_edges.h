#ifndef _ADJ_EDGES_H_
#define _ADJ_EDGES_H_

#include <vector>

class AdjEdges
{
private:
    int vertices;
    int entries;
    std::vector<std::vector<int>> data;
public:
    AdjEdges(const char* path);
    ~AdjEdges();

    int num_vertices() const;
    int num_entries() const;
    const std::vector<int>& operator[](int index) const;
};

#endif