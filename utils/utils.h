#ifndef UTILS_H
#define UTILS_H


float max_diff(float *res1, float *res2, int n);
int n_zeros(float *a, int n);
void fill_array(float *a, int n);
void test_res(float *res1, float *res2, int n);
void print_array(float *a, int n);
void init_zero(float *a, int n);
void set_eq(float *a, float *b, int n);
void kaiming_init(float *w, int n_in, int n_out);
int random_int(int min, int max);

/*
    find out the index of max number in an array and store it in the location pointed by result.
    inputs:
            arr       -------------- pointer to the arrary which contain the input data
            arrl      -------------- the length of the array pointed by arr
            segment   --------------

*/
void max_element_index(float* const arr, int arrl,int * const result);

#endif
