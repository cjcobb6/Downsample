#include "boost/multi_array.hpp"
#include <iostream>
#include <unordered_map>
#include <thread>
#include <algorithm>
#include <math.h>

struct Block
{
    int val;
    std::unordered_map<int,size_t> frequencies;

    Block(int v, std::unordered_map<int,size_t> f) : 
        val(v), 
        frequencies(f) 
    {}

    Block() {}
};


void printBlock(Block b) 
{
    std::cout << "val= " << b.val;
    std::cout << " . frequencies= [";
    for(std::unordered_map<int,size_t>::iterator it = b.frequencies.begin(); it != b.frequencies.end(); ++it) 
    {
        std::cout << it->first << ":" << it->second << " , "; 
    }
    std::cout << std::endl;
}

std::vector<size_t> vectorizeStrides(
        const boost::multi_array_types::index* strides,
        size_t num_strides)
{
    std::vector<size_t> strides_vec; 
    for(size_t i = 0; i < num_strides; ++i)
    {
        strides_vec.push_back(strides[i]);
    }
    return strides_vec;
}

template <unsigned long N>
std::vector<Block> flatten(boost::multi_array<int, N>& image)
{
    std::vector<Block> blocks;
    for(int* ptr = image.data(); ptr < image.data() + image.num_elements(); ++ptr) 
    {
        std::unordered_map<int,size_t> freq;
        freq[*ptr] = 1;
        blocks.emplace_back(*ptr,freq);
    }
    return blocks;
}

//merges frequencies in block into cur_frequencies. used while building block
void mergeBlock(Block& b, std::unordered_map<int, size_t>& cur_frequencies)
{
    for(std::unordered_map<int, size_t>::iterator it = b.frequencies.begin();
            it != b.frequencies.end(); ++it) 
    {
        if(cur_frequencies.find(it->first) == cur_frequencies.end()) 
        {
            cur_frequencies[it->first] = it->second;
        }
        else
        {
            cur_frequencies[it->first] = 
                cur_frequencies[it->first] + it->second;
        }   
    }
}

//given a starting block, traverse adjacent blocks and populate frequency map
void traverse(
        std::vector<Block>& blocks,
        size_t levels_to_traverse,
        size_t index,
        std::vector<size_t>& strides,
        std::unordered_map<int,size_t>& frequencies)
{
    mergeBlock(blocks[index], frequencies);
    for(size_t i = 0; i < levels_to_traverse; ++i) {
        size_t stride = strides[strides.size() - 1 - i];
        traverse(blocks, i, index + stride, strides, frequencies);
    }
}

//build a block out of all blocks adjacent to blocks[index]
Block makeBlock(
        std::vector<Block>& blocks,
        size_t levels_to_traverse,
        size_t index,
        std::vector<size_t>& strides)
{
    std::unordered_map<int, size_t> frequencies;
    traverse(blocks, levels_to_traverse, index, strides, frequencies);
    int max_val = 0;
    int max_freq = 0;
    for(std::unordered_map<int,size_t>::iterator it = frequencies.begin(); 
            it != frequencies.end(); ++it)
    {
        if(it == frequencies.begin() || it->second > max_freq) 
        {
            max_val = it->first;
            max_freq = it-> second;
        }
    }

    Block b;
    b.frequencies = frequencies;
    b.val = max_val;
    return b;
}

size_t getNextIndex(size_t index, std::vector<size_t>& strides)
{
    std::vector<bool> strides_used;
    strides_used.resize(strides.size());
    for(size_t i = 0; i < strides.size(); ++i)
    {
        strides_used.push_back(false);
    }
    index += 2;
    for(size_t iter = 0; iter < strides.size() - 1; ++iter){
        for(size_t i = 0; i < strides.size() - 1; ++i)
        {
            //at end of row/column/whatever
            if(index % strides[i] == 0 && !strides_used[i]) 
            {
                //skip next row
                index += strides[i];
                strides_used[i] = true;
                break;
            }
        }
    }
    return index;
}

//gets all indices of base blocks used for downsampling
//length of indices will be size of downsampled image
std::vector<size_t> getIndexVector(std::vector<size_t>& strides, size_t max)
{
    std::vector<size_t> indices;
    size_t index = 0;
    while(index < max) 
    {
        indices.push_back(index);
        index = getNextIndex(index, strides);
    }
    return indices;
}

//used in concurrent downsample
//new_blocks is populated concurrently by worker threads
//each thread has an offset, which signifies which indices
//that worker is responsible for
//i.e. thread i works on every index such that index % num_threads == i
void populateBlocks(
        std::vector<Block>& new_blocks, 
        size_t start, 
        size_t num_threads,
        std::vector<Block>& old_blocks,
        std::vector<size_t>& indices,
        std::vector<size_t>& strides) 
{
    size_t dimensions = strides.size();
    for(size_t i = start; i < indices.size(); i += num_threads)
    {
        size_t index = indices[i];
        new_blocks[i] = makeBlock(old_blocks, dimensions, index, strides);
    }
}


//return 1-downsampled image of orig_blocks
std::vector<Block> downsampleConcurrent(
        std::vector<Block>& old_blocks,
        std::vector<size_t>& strides,
        size_t num_threads=2)
{
    std::vector<Block> new_blocks;
    std::vector<size_t> indices = getIndexVector(strides, old_blocks.size());
    new_blocks.resize(indices.size());

    std::vector<std::thread*> workers;
    for(size_t i = 0; i < num_threads; ++i)
    {
        std::thread* t = new std::thread(populateBlocks, 
                std::ref(new_blocks),
                i,
                num_threads,
                std::ref(old_blocks),
                std::ref(indices),
                std::ref(strides));
        workers.push_back(t);
    } 
    for(size_t i = 0; i < num_threads; ++i)
    {
        workers[i]->join();
        delete workers[i];
    }
    return new_blocks;
}

//return 1-downsampled image of orig_blocks
std::vector<Block> downsample(
        std::vector<Block>& orig_blocks,
        std::vector<size_t>& strides)
{
    std::vector<Block> new_blocks;
    size_t dimensions = strides.size();
    size_t index = 0;
    while(index < orig_blocks.size()) 
    {
        Block b = makeBlock(orig_blocks, dimensions, index, strides);
        new_blocks.push_back(b);
        index = getNextIndex(index, strides);
    }
    return new_blocks;
}

template <unsigned long N>
size_t getMaxPossibleDownsample(boost::multi_array<int, N>& image) 
{
    size_t dimensions = image.num_dimensions();
    const size_t* shape = image.shape();
    size_t min;
    for(size_t i = 0; i < dimensions; ++i)
    {
        if(i == 0 || shape[i] < min)
        {
            min = shape[i];
        }
    }
    return log2(min);
}

//obtain strides of downsampled image
std::vector<size_t> getNewStrides(std::vector<size_t>& strides) 
{
    std::vector<size_t> new_strides;
    for(size_t i = 0; i < strides.size(); ++i)
    {
        unsigned long one = 1;
        unsigned long divisor = std::pow(2, strides.size() - i - 1);
        new_strides.push_back(std::max(one,strides[i]/divisor));
    }
    return new_strides;
}
template <unsigned long N>
std::vector<std::vector<Block> > getAllDownsamplingsConcurrent(
        boost::multi_array<int, N> image)
{
    std::vector<std::vector<Block>> blocks;
    std::vector<Block> prev_iter_blocks = flatten(image);

    std::vector<size_t> strides = 
        vectorizeStrides(image.strides(), image.num_dimensions());
    size_t max_downsample = getMaxPossibleDownsample(image);
    for(size_t i = 0; i < max_downsample; ++i) 
    {
        prev_iter_blocks = downsampleConcurrent(prev_iter_blocks, strides);
        blocks.push_back(prev_iter_blocks);
        strides = getNewStrides(strides);
    }
    return blocks;
}

template <unsigned long N>
std::vector<std::vector<Block> > getAllDownsamplings(
        boost::multi_array<int, N> image)
{
    std::vector<std::vector<Block>> blocks;
    std::vector<Block> prev_iter_blocks = flatten(image);

    std::vector<size_t> strides = 
        vectorizeStrides(image.strides(), image.num_dimensions());
    size_t max_downsample = getMaxPossibleDownsample(image);
    blocks.reserve(max_downsample);
    for(size_t i = 0; i < max_downsample; ++i) 
    {
        prev_iter_blocks = downsample(prev_iter_blocks, strides);
        blocks.push_back(prev_iter_blocks);
        strides = getNewStrides(strides);
    }
    return blocks;
}

template <unsigned long N>
void printImage(boost::multi_array<int, N> arr)
{
    std::cout << "printing image" << std::endl;
    for(int* i = arr.origin(); i < (arr.origin() + arr.num_elements()); ++i)
    {
        std::cout << *i << " , ";
    }
    std::cout << std::endl;
}

std::vector<std::vector<int> > toInts(std::vector<std::vector<Block> >& blocks)
{
    std::vector<std::vector<int> > vecs;
    for(size_t i = 0; i < blocks.size(); ++i)
    {
        std::vector<int> vec;

        for(size_t j = 0; j < blocks.size(); ++j)
        {
            vec.push_back(blocks[i][j].val);
        }
        vecs.push_back(vec);
    }
    return vecs;
}


int main(int argc, char** argv) 
{
    std::cout << "running tests" << std::endl;

    std::cout << "2 dimensional square test..." << std::endl;
    boost::multi_array<int,2> A(boost::extents[4][4]);
    A[0][0]=1;
    A[0][1]=1;
    A[1][0]=1;
    A[3][0]=2;
    A[3][1]=2;
    A[2][0]=1;
    std::vector<std::vector<Block> > downsamplings = getAllDownsamplings(A);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "2 dimensional square test concurrent..." << std::endl;
    downsamplings = getAllDownsamplingsConcurrent(A);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "2 dimensional rectangle test..." << std::endl;
    boost::multi_array<int,2> B(boost::extents[4][2]);
    B[0][0]=1;
    B[0][1]=1;
    B[1][0]=1;
    B[3][0]=2;
    B[3][1]=2;
    B[2][0]=1;
    downsamplings = getAllDownsamplings(B);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "2 dimensional rectangle test concurrent..." << std::endl;
    downsamplings = getAllDownsamplingsConcurrent(B);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "3 dimensional square test..." << std::endl;
    boost::multi_array<int,3> C(boost::extents[2][2][2]);
    C[0][0][0]=1;
    downsamplings = getAllDownsamplings(C);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "3 dimensional square test concurrent..." << std::endl;
    C[0][0][0]=1;
    downsamplings = getAllDownsamplingsConcurrent(C);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
     
    }

    std::cout << "3 dimensional bigger square test..." << std::endl;
    boost::multi_array<int,3> D(boost::extents[4][4][4]);
    D[0][0][0]=1;
    downsamplings = getAllDownsamplings(D);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "3 dimensional bigger square test concurrent..." << std::endl;
    downsamplings = getAllDownsamplingsConcurrent(D);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "3 dimensional rectanle..." << std::endl;
    boost::multi_array<int,3> E(boost::extents[4][4][2]);
    E[0][0][0]=1;
    E[0][0][1]=1;
    E[0][1][0]=1;
    E[0][1][1]=1;
    E[1][0][0]=1;
    downsamplings = getAllDownsamplings(E);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "3 dimensional rectanle concurrent..." << std::endl;
    downsamplings = getAllDownsamplingsConcurrent(E);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "4 dimensional concurrent" << std::endl;
    boost::multi_array<int,4> F(boost::extents[4][4][4][4]);
    downsamplings = getAllDownsamplingsConcurrent(F);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    std::cout << "4 dimensional big concurrent" << std::endl;
    boost::multi_array<int,4> G(boost::extents[16][16][16][16]);
    downsamplings = getAllDownsamplingsConcurrent(G);
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }

    return 0;
}
