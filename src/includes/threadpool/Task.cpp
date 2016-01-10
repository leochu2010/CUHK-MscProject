#include "Task.h"
#include <stdlib.h>

Task::Task(void (*fn_ptr)(void*), void* arg) : m_fn_ptr(fn_ptr), m_arg(arg) {
}

Task::~Task() {
}

void Task::operator()() {
  (*m_fn_ptr)(m_arg);
  if (m_arg != NULL) {	  
    //delete m_arg;
	free(m_arg);
  }
}

void Task::run() {
  (*m_fn_ptr)(m_arg);
}
