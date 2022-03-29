#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "adj_edges.h"

/**
 * NOTE: vertices in dataset start from index 1.
 */
AdjEdges::AdjEdges(const char* path) {
    vertices = 0;
    entries = 0;
    std::ifstream ifile(path);
    std::string str1, str2;
    int count = 0;
    while (ifile >> str1 >> str2)
    {
        int num1 = std::stoi(str1);
        int num2 = std::stoi(str2);
        vertices = std::max(vertices, std::max(num1, num2));
        data.push_back({num1 - 1, num2 - 1});
        entries++;
    }
    ifile.close();
}

AdjEdges::~AdjEdges() {
}

int AdjEdges::num_vertices() const {
    return vertices;
}

int AdjEdges::num_entries() const {
    return entries;
}

const std::vector<int>& AdjEdges::operator[](int index) const {
    return data[index];
}