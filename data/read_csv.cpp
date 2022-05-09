#include <fstream>

#include "read_csv.h"


void read_csv(float *inp, std::string name,int totalline){
    std::ifstream file(name);
    std::string line;
    int linecount=0;
    while(std::getline(file, line, '\n') && (linecount<totalline)){
        *inp = std::stof(line);
        inp++;
        linecount++;
    }
}
