#include <iostream>
#include <fstream>

#include "adj_list.h"

AdjList::AdjList(const char* path) {
    std::ifstream ifile(path);
    std::string str1, str2;
    int count = 0;
    while (ifile >> str1 >> str2)
    {
        int num1 = std::stoi(str1);
        int num2 = std::stoi(str2);
        vertices = std::max(vertices, std::max(num1, num2));
        data[num1 - 1].insert(num2 - 1);
        data[num2 - 1].insert(num1 - 1);
    }
    ifile.close();
    for (int i = 0; i < vertices; i++) {
        edges += data[i].size();
    }
}

AdjList::~AdjList()
{
}

int AdjList::num_vertices() const {
    return vertices;
}

int AdjList::num_edges() const {
    return edges;
}