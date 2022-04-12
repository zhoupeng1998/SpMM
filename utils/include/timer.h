#ifndef _TIMER_H_
#define _TIMER_H_

void clock_start_cpu();
void clock_stop_cpu();
double get_time_cpu();

void clock_start_cuda();
void clock_stop_cuda();
double get_time_cuda();

#endif