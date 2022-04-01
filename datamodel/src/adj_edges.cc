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

int AdjEdges::CountNNZ() const {
    int count=0;
    for(auto edge:data){
        count++;
    }
    return count;
}

int AdjEdges::CountRows() const {
    int count=0;
    int previous = 0;

    for(auto edge:data){
        if(edge[1]!=previous){
            count++;
            previous=edge[1];
        }

    }
    return count;
}



int AdjEdges::num_entries() const {
    return entries;
}

const std::vector<int>& AdjEdges::operator[](int index) const {
    return data[index];
}