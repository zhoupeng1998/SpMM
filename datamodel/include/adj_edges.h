#ifndef _ADJ_EDGES_H_
#define _ADJ_EDGES_H_

#include <vector>

#include "data.h"

class AdjEdges
{
private:
    int vertices;
    int entries;


public:
    std::vector<std::vector<int> > data;
    AdjEdges(const char* path);
    AdjEdges(const char* path, int limit);
    ~AdjEdges();
    int CountNNZ() const;
    int num_vertices() const;
    int CountRows() const;
    int num_entries() const;
    const std::vector<int>& operator[](int index) const;
};

#endif