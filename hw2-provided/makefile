CC=gcc
CFLAG= -Wall -I. -O0 -fopenmp  # Added -fopenmp for OpenMP support

TARGETS=unoptimized optimized # add your target here

all: $(TARGETS)

unoptimized: unoptimized.o microtime.o
	$(CC) -o $@ $^

unoptimized.o: unoptimized.c microtime.h
	$(CC) $(CFLAG) -c $<

microtime.o: microtime.c microtime.h
	$(CC) $(CFLAG) -c $<

# Add your rules here.
# Example: code in opt1.c

optimized: optimized.o microtime.o
	$(CC) $(CFLAG) -o $@ $^

optimized.o: optimized.c microtime.h
	$(CC) $(CFLAG) -c $<


clean:
	rm -f *.o *~ core $(TARGETS)
