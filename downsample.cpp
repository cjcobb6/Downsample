#include "boost/multi_array.hpp"
#include <iostream>
#include <unordered_map>
#include <thread>
#include <algorithm>
#include <math.h>
typedef boost::multi_array<int, 3> array_type;
typedef array_type::index array_index_type;
/*

//N must be specified at compile time
template <int N>
void foo(boost::multi_array<int, N> n_dimensional_arr)
{
    int num_dimensions = N;

    std::cout << n_dimensional_arr.size() << std::endl;
}

template <unsigned long N>
int getFirstElt(boost::multi_array<int, N> n_dimensional_arr)
{
    int* start =  n_dimensional_arr.data();
    int* curr = start;
    const boost::multi_array_types::index* strides = n_dimensional_arr.strides();
    for(size_t i = 0; i < n_dimensional_arr.num_dimensions(); ++i)
    {
        curr = curr + strides[i];
    }
    return *curr;
}

void printStrides(const boost::multi_array_types::index* strides, size_t size)
{
    std::cout << "printing strides" << std::endl;
    for(size_t i = 0; i < size; ++i)
    {
        std::cout << strides[i] << " , ";
    }
    std::cout << std::endl;
}

void increment(int val, std::unordered_map<int,size_t>& freq)
{
    if(freq.find(val) == freq.end()) 
    {
        freq[val] = 1;
    }
    else
    {
        freq[val] = freq[val] + 1;
    }

}

template <unsigned long N>
std::unordered_map<int, size_t> getFirstImage(boost::multi_array<int, N> n_dimensional_arr, int offset=0)
{
    std::cout << "getting image" << std::endl;
    std::unordered_map<int, size_t> frequencies;
    const boost::multi_array_types::index* strides = n_dimensional_arr.strides();
    int* start =  n_dimensional_arr.data();
    int* curr = start;
    if(offset > 0) {
        curr = curr + 1 + strides[0] * offset;
    }
    //std::cout << "num cells in block " << pow(2,n_dimensional_arr.num_dimensions()) << std::endl;
    size_t j = n_dimensional_arr.num_dimensions();
    for(size_t i = 0; i < pow(2,n_dimensional_arr.num_dimensions()); ++i)
    {
        increment(*(curr+i), frequencies);
        double int_part;
        if(i % 2 == 1) 
        {
            curr = start + strides[j];
            j--;
        } 
    }
    return frequencies;
}

*/



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

//i like this
void traverse(
        std::vector<Block>& blocks,
        size_t levels_to_traverse,
        size_t index,
        std::vector<size_t>& strides,
        std::unordered_map<int,size_t>& frequencies)
{
   // std::cout << "index is : " << index << std::endl;
    mergeBlock(blocks[index], frequencies);
    for(size_t i = 0; i < levels_to_traverse; ++i) {
        size_t stride = strides[strides.size() - 1 - i];
        traverse(blocks, i, index + stride, strides, frequencies);
    }
}

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

void populateBlocks(
        std::vector<Block>& new_blocks, 
        size_t start, 
        size_t num_threads,
        std::vector<Block>& old_blocks,
        std::vector<size_t>& indices,
        std::vector<size_t>& strides) 
{
    size_t dimensions = strides.size();
    //std::cout << "running populate blocks" << std::endl;
    for(size_t i = start; i < indices.size(); i += num_threads)
    {
        size_t index = indices[i];
      //  std::cout << "index is " << index << std::endl;
        new_blocks[i] = makeBlock(old_blocks, dimensions, index, strides);
    }
}


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

std::vector<Block> downsample(
        std::vector<Block>& orig_blocks,
        std::vector<size_t>& strides)
{
    std::vector<Block> new_blocks;
    size_t dimensions = strides.size();
    size_t index = 0;
    //std::cout << "orig_blocks size is " << orig_blocks.size() << std::endl;
    while(index < orig_blocks.size()) 
    {
      //  std::cout << "making block" << std::endl;
        Block b = makeBlock(orig_blocks, dimensions, index, strides);
        //std::cout << "made block" << std::endl;
        new_blocks.push_back(b);
        index = getNextIndex(index, strides);
        //std::cout << "next index is : " << index << std::endl;
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
std::vector<std::vector<Block> > getAllDownsamplingsConcurrent(boost::multi_array<int, N> image)
{
    std::vector<std::vector<Block>> blocks;
    //std::cout << "flattening" << std::endl;
    std::vector<Block> prev_iter_blocks = flatten(image);

    //std::cout << "vectorizing" << std::endl;
    std::vector<size_t> strides = 
        vectorizeStrides(image.strides(), image.num_dimensions());
    //std::cout << "getting max" << std::endl;
    size_t max_downsample = getMaxPossibleDownsample(image);
    //std::cout << "max is " << max_downsample << std::endl;
    for(size_t i = 0; i < max_downsample; ++i) 
    {
      //  std::cout << "downsampling on iteration " << i << std::endl;
        prev_iter_blocks = downsampleConcurrent(prev_iter_blocks, strides);
        blocks.push_back(prev_iter_blocks);
        strides = getNewStrides(strides);
    }
    return blocks;
}

template <unsigned long N>
std::vector<std::vector<Block> > getAllDownsamplings(boost::multi_array<int, N> image)
{
    std::vector<std::vector<Block>> blocks;
    //std::cout << "flattening" << std::endl;
    std::vector<Block> prev_iter_blocks = flatten(image);

    //std::cout << "vectorizing" << std::endl;
    std::vector<size_t> strides = 
        vectorizeStrides(image.strides(), image.num_dimensions());
    //std::cout << "getting max" << std::endl;
    size_t max_downsample = getMaxPossibleDownsample(image);
    //std::cout << "max is " << max_downsample << std::endl;

    for(size_t i = 0; i < max_downsample; ++i) 
    {
        for(size_t i = 0; i < strides.size(); ++i) 
        {
            //std::cout << strides[i] << std::endl;
        }
        //std::cout << "downsampling on iteration " << i << std::endl;
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



    std::cout << "2 dimensional square test..." << std::endl;
    boost::multi_array<int,2> A(boost::extents[4][4]);
    A[0][0]=1;
    A[0][1]=1;
    A[1][0]=1;
    A[3][0]=2;
    A[3][1]=2;
    A[2][0]=1;
    std::vector<std::vector<Block> > downsamplings = getAllDownsamplings(A);
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
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
    //need to check number of elements. is size correct?
    //std::cout <<  "number of elements " << A.num_elements() << std::endl;
    for(size_t i = 0; i < downsamplings.size(); ++i) 
    {
        for(size_t j = 0; j < downsamplings[i].size(); ++j) 
        {
            std::cout << downsamplings[i][j].val << " , ";
        }
        std::cout << std::endl;
    }
    
    return 0;
        /*
    printImage(simple);
    printStrides(simple.strides(), simple.num_dimensions());
    std::unordered_map<int, size_t> map = getBlock(simple);
    for(std::unordered_map<int, size_t>::iterator it = map.begin(); it != map.end(); ++it) 
    {

        std::cout << it->first << " : " << it->second << std::endl;
    }
    std::cout << "*********************" << std::endl;
    boost::multi_array<int,2> A(boost::extents[4][8]);

    std::cout << A.size() << std::endl;
    std::cout << A[0].size() << std::endl;
    A[0][0] = 1;
    A[1][0] = 1;
    boost::multi_array<int,3> B(boost::extents[2][2][2]);
    B[0][0][0] = 1;
    B[0][0][1] = 1;
    B[0][1][0] = 3;
    B[0][1][1] = 1;
    boost::multi_array<int,4> C(boost::extents[2][2][2][2]);


    printImage(A);
    printStrides(A.strides(), A.num_dimensions());
    map = getBlock(A,A.strides()[0]);
    for(std::unordered_map<int, size_t>::iterator it = map.begin(); it != map.end(); ++it) 
    {
        std::cout << it->first << " : " << it->second << std::endl;
    }
    std::cout << "*********************" << std::endl;
    printImage(B);
    printStrides(B.strides(), B.num_dimensions());
    map = getBlock(B);
    for(std::unordered_map<int, size_t>::iterator it = map.begin(); it != map.end(); ++it) 
    {
        std::cout << it->first << " : " << it->second << std::endl;
    }
    std::cout << "*********************" << std::endl;
    printImage(C);
    printStrides(C.strides(), C.num_dimensions());
    map = getBlock(C);
    for(std::unordered_map<int, size_t>::iterator it = map.begin(); it != map.end(); ++it) 
    {
        std::cout << it->first << " : " << it->second << std::endl;
    }
    std::cout << "*********************" << std::endl;
*/
    return 0;
}



/*
template <int N>
class BlockMaker
{
    private:
        size_t m_numDimensions;
        std::unordered_map<int, size_t> m_frequencies;
        std::vector<boost::multi_array_types::index> m_strides;
        int m_curMax;
        size_t m_curMaxFreq;
        bool m_curMaxSet;
        size_t m_maxStrideIndex;
        size_t m_imageSize;
        std::unordered_map<size_t,bool> m_indexMap;
        int* m_start;

        std::vector<Block> m_blocks;
        std::vector<bool> m_visited;

        int getNextIndex()
        {
            for(size_t i = 0; i < m_visited; ++i) 
            {
                if(!m_visited[i]) 
                {
                    return i;
                }
            }
            return -1;
        }

        void makeVisitedVector() 
        {
            for(int i = 0; i < m_imageSize; ++i) 
            {
                m_visited[i] = false;
            }
        }

        void flatten(boost::multi_array<int, N> image)
        {
            for(int* ptr = image.data(); ptr < image.data() + image.num_elements(); ++ptr) 
            {
                std::unordered_map<int,size_t> freq;
                freq[*ptr] = 1;
                m_blocks.emplace_back(*ptr,freq);
            }
        }
    public:


        BlockMaker(boost::multi_array<int, N> image) :
            m_start(image.data()),
            m_numDimensions(image.num_dimensions()),
            m_curMax(0),
            m_curMaxSet(false),
            m_imageSize(image.num_elements())

    {
        m_strides = vectorizeStrides(image.strides(), image.num_dimensions());
        makeIndexVector();
        flatten(image);
        combineBlocks();
        makeBlocks();
    }

        BlockMaker(
                std::vector<Block> blocks,
                size_t num_dimensions,
                std::vector<boost::multi_array_types::index> strides, 
                size_t max_stride_index,
                size_t image_size) :
            m_start(0),
            m_numDimensions(num_dimensions),
            m_curMax(0),
            m_curMaxSet(false),
            m_maxStrideIndex(max_stride_index),
            m_imageSize(image_size),
            m_strides(strides)
    {
        makeBlocksFromBlocks(blocks);
    }   

        void combineBlocks()
        {

            int levels_to_traverse = m_numDimensions;
            int current_index = 0;
            for(int idx = 0; idx != -1; idx = getNextIndex()) 
            {
                traverse(current_block, levels_to_traverse);
                m_blocks.emplace_back(m_curMax,m_frequencies);
                reset();
            }
        }

        void makeBlocks2()
        {

            int levels_to_traverse = m_numDimensions;
            int* current_block = m_start;
            for(size_t i = 0; i < (m_imageSize / std::pow(2,m_numDimensions)); ++i)
            {
                traverse(current_block + getJumpDistance(m_strides)*i,levels_to_traverse);
                Block b(m_curMax, m_frequencies);
                printBlock(b);
                m_blocks.emplace_back(m_curMax,m_frequencies);
                reset();
            }
        }

        int getJumpDistance(std::vector<boost::multi_array_types::index>& strides) 
        {

            int jumpDistance = strides[0]+1;
            for(size_t i = strides.size()-1; i > 0; --i)
            {
                if(strides[i] + 1 != strides[i-1])
                {
                    jumpDistance = strides[i] + 1;
                    break;
                }
            }
            return jumpDistance;
        } 


        void makeBlocksFromBlocks(std::vector<Block> blocks)
        {

            int levels_to_traverse = m_numDimensions;
            int* current_block = m_start;
            for(size_t i = 0; i < (m_imageSize / std::pow(2,m_numDimensions)); ++i)
            {
                std::cout << "about to traverse" << std::endl;
                traverse(blocks, levels_to_traverse, i * getJumpDistance(m_strides));
                m_blocks.emplace_back(m_curMax,m_frequencies);
                reset();
            }
        }

        //i like this
        void traverse(
                int* curr,
                size_t levels_to_traverse)
        {
            std::cout << curr - m_start << std::endl;
            increment(*curr);
            m_indexMap[curr - m_start] = true;
            for(size_t i = 0; i < levels_to_traverse; ++i) {
                size_t stride = m_strides[m_numDimensions - i - 1];
                traverse(curr + stride, i);
            }
        }

        //i like this
        void traverse(
                std::vector<Block>& blocks,
                size_t levels_to_traverse,
                size_t index)
        {

            increment(blocks[index]);
            std::cout << index << std::endl;
            for(size_t i = 0; i < levels_to_traverse; ++i) {
                size_t stride = m_strides[m_numDimensions - i - 1];
                traverse(blocks, i, index + stride);
            }
        }

        void increment(int val)
        {
            if(m_frequencies.find(val) == m_frequencies.end()) 
            {
                m_frequencies[val] = 1;
            }
            else
            {
                m_frequencies[val] = m_frequencies[val] + 1;
            }


            std::cout << "frequency of " << val << " is " << m_frequencies[val] << std::endl;
            if(!m_curMaxSet || m_frequencies[val] > m_frequencies[m_curMax])
            {
                m_curMax = val;
                m_curMaxSet = true;
            }

        }

        void increment(Block b)
        {
            m_frequencies = mergeFrequencies(m_frequencies, b.frequencies);
        }

        std::unordered_map<int, size_t> mergeFrequencies(
                std::unordered_map<int, size_t> freq1,
                std::unordered_map<int, size_t> freq2)
        {
            std::unordered_map<int, size_t> merged;
            for(std::unordered_map<int, size_t>::iterator it = freq1.begin(); it != freq1.end(); ++it)
            {
                merged[it->first] = it->second;
                if(it->second > m_frequencies[m_curMax]) 
                {
                    m_curMax = it->first;
                }
            }

            for(std::unordered_map<int, size_t>::iterator it = freq2.begin(); it != freq2.end(); ++it)
            {
                if(merged.find(it->first) == merged.end()) 
                {
                    merged[it->first] = it->second;
                } else 
                {
                    merged[it->first] = merged[it->first] + it->second;
                }
                if(!m_curMaxSet || merged[it->first] > m_frequencies[m_curMax]) 
                {
                    m_curMax = it->first;
                    m_curMaxSet = true;
                }
            }
            return merged;
        }



        void reset() 
        {
            m_curMaxSet = false;
            m_frequencies.clear();
        }

        std::vector<Block> getBlocks()
        {

            int levels_to_traverse = m_numDimensions;
            return m_blocks;
        }
};//end BlockMaker class

template <unsigned long N>
size_t getMaxPossibleDownsample(boost::multi_array<int, N> image)
{

    const size_t* sizes = image.shape();
    size_t maxPossible = -1;
    for(size_t i = 0; i < N; ++i)
    {
        if(maxPossible == -1 || sizes[i] < maxPossible) 
        {
            maxPossible = sizes[i];
        }
    }
    return std::log2(maxPossible);
}



 std::vector<boost::multi_array_types::index> fixStrides(std::vector<boost::multi_array_types::index> strides)
{
    std::vector<boost::multi_array_types::index> new_strides; 
    for(size_t i = 1; i < strides.size(); ++i) 
    {
        new_strides.push_back(strides[i]/2);
    }
    return new_strides;
}


std::vector<std::vector<int> > getDownsamplingsFromBlocks(std::vector<std::vector<Block> > blocks)
{
    std::vector<std::vector<int> > downsamplings;  
    for(size_t i = 0; i < blocks.size(); ++i) 
    {
        std::vector<int> downsample;
        for(size_t j = 0; j < blocks[i].size(); ++j) 
        {
            downsample.push_back(blocks[i][j].val);
        } 
        downsamplings.push_back(downsample);
    }
    return downsamplings;
} 


void printBlocks(std::vector<Block> blocks) 
{
    for(size_t i = 0; i < blocks.size(); ++i)
    {
        printBlock(blocks[i]);
    }
}





template <unsigned long N>
std::vector<std::vector<int> > getDownsamplings(boost::multi_array<int, N>& image) 
{
    std::vector<std::vector<Block> > downsamplings;
    size_t num_strides = image.num_dimensions();
    size_t cur_image_size = image.num_elements();
    size_t max_possible_downsample = getMaxPossibleDownsample(image);
    std::vector<boost::multi_array_types::index> strides = vectorizeStrides(image.strides(), num_strides);
    BlockMaker<N> block_maker(image);
    for(size_t i = 0; i < max_possible_downsample; ++i) 
    {
        if(i > 0)
        {
            block_maker = BlockMaker<N>(downsamplings[i-1], image.num_dimensions(),
            strides, i, cur_image_size);
        }
        std::vector<Block> blocks = block_maker.getBlocks();
        printBlocks(blocks);
        downsamplings.push_back(blocks);
        cur_image_size = cur_image_size / std::pow(2, image.num_dimensions());
        strides = fixStrides(strides);
    }
    return getDownsamplingsFromBlocks(downsamplings);
    //make blocks with blockmaker
    //transform blocks into raw bytes with just modes, push into vector
    //pop front off strides
    //make blocks out of new blocks using new strides
    //repeat until last strides is popped
}


//i like this
void traverse(
        int* curr,
        size_t num_dimensions,
        const boost::multi_array_types::index* strides,
        size_t levels_to_traverse,
        std::unordered_map<int, size_t>& frequencies)
{

    increment(*curr, frequencies);
    for(size_t i = 0; i < levels_to_traverse; ++i) {
        size_t stride = strides[num_dimensions - i - 1];
        traverse(curr + stride, num_dimensions, strides, i, frequencies);
    }
}


template <unsigned long N>
std::unordered_map<int, size_t> getBlock(boost::multi_array<int, N> n_dimensional_arr, int offset=0)
{
    std::cout << "getting block" << std::endl;
    std::unordered_map<int, size_t> frequencies;
    size_t levels_to_traverse = n_dimensional_arr.num_dimensions();
    size_t num_dimensions = n_dimensional_arr.num_dimensions();
    const boost::multi_array_types::index* strides = n_dimensional_arr.strides();
    int* curr = n_dimensional_arr.data() + offset;
    traverse(curr, num_dimensions, strides, levels_to_traverse, frequencies);
    return frequencies;
}



*/
