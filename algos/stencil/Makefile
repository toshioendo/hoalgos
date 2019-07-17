all: OMPFOR

APP=stencil

OMPFOR:
	make -f Makefile.OMPFOR

OMPTASK:
	make -f Makefile.OMPTASK

CUDAUM:
	make -f Makefile.CUDAUM

MCUDAUM:
	make -f Makefile.MCUDAUM

clean:
	rm -f *.o
	rm -f *~
	rm -f core
	rm -f core.*
	rm -f $(APP)
	rm -f a.out
