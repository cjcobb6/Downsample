Explanation of algorithm

My algorithm takes a boost multi array and transforms it into a flat array of
Blocks, a class I defined. These blocks contain the integers they represent,
as well as the frequency of characters of the blocks that were merged together
to form said block. This allows us to pass the previous iteration's blocks
into the next without having to start from scratch, while still
preserving the frequencies from the original image.

The concurrent part of my algorithm is lock free and wait free in each
iteration, as each thread operates entirely indendently of the other threads.
Since we feed the previous iterations result to the next, we must wait in the
main thread for each iteration to finish.

Analysis

Let N be the number of pixels in the image, n be the number of unique pixels
and M the number of threads

The execution time of my algorithm, assuming O(1) hashmap access is O(N/M)
Explanation:
In each downsampling iteration, each cell is touched exactly once
The 1-downsampled image of image i has less than half the number of cells of
image i
This is because the length of the array in each dimension is cut in half
The summation N + N/2 + N/4 + ... + 2 = O(N)
Therefore, my algorithm is O(N) in the single threaded case
With M threads, each thread does an equal amount of work and no cell is touched
more than once, resulting in O(N/M)
If we assume worst case hashmap access, which would be O(n),
the exection time of our algorithm is O(N/M * n)

The memory usage of my algorithm is O(N), for similar reasons as above
The first image is size N, the second is less than N/2, and so on
The frequencies map will use O(n) memory, but n can't be bigger than N
The number of threads does not affect memory usage scaling, as each thread
operates independently.


