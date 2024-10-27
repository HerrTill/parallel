CC=gcc
CFLAG= -Wall -I. -O0 -fopenmp

TARGETS=unoptimized optimized intel  # Add your target here

all: $(TARGETS)

# Rule for unoptimized target
unoptimized: unoptimized.o microtime.o
	$(CC) -o $@ $^

unoptimized.o: unoptimized.c microtime.h
	$(CC) $(CFLAG) -c $<

# Rule for microtime object
microtime.o: microtime.c microtime.h
	$(CC) $(CFLAG) -c $<

# Rule for optimized target
optimized: optimized.o microtime.o
	$(CC) -o $@ $^

optimized.o: optimized.c microtime.h
	$(CC) $(CFLAG) -c $<

# Rule for intel target with AVX instructions
intel: intel.o microtime.o
	$(CC) -mavx -o $@ $^

# Rule for compiling intel object file with AVX support
intel.o: intel.c microtime.h
	$(CC) $(CFLAG) -mavx -c $<  # Added -c flag to compile only

# Rule for cleaning up
clean:
	rm -f *.o *~ core $(TARGETS)
