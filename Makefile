NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -x cu --std=c++11 -arch=sm_20 -rdc=true -g

GCC = gcc
GCCFLAGS = -Wall --std=c99 -g

LINKFLAGS = --std=c++11 -arch=sm_20 -g

EXE = simulation

BOLD = \\033[1m
NORMAL = \\033[0m
GREEN = \\033[1;32m



define n


endef

all: dist/$(EXE)

dist/${EXE}: obj/cu obj/c dist .linking
	
obj/c: obj
	@printf "${GREEN} - make obj/c folder${NORMAL}\n"
	mkdir -p obj/c

obj/cu: obj
	@printf "${GREEN} - make obj/cu folder${NORMAL}\n"
	mkdir -p obj/cu
	
obj:
	@printf "${GREEN} - make obj folder${NORMAL}\n"
	mkdir -p obj
	
dist:
	@printf "${GREEN} - make dist folder${NORMAL}\n"
	mkdir -p dist

.linking: $(patsubst src%, obj%, $(patsubst %.cu, %.o, $(patsubst %.c, %.o,$(shell find src -type f))))
	@printf "${GREEN} - link program from objects${NORMAL}\n"
	$(NVCC) $(LINKFLAGS) $(patsubst src%, obj%, $(patsubst %.cu, %.o, $(patsubst %.c, %.o, $^))) -o dist/$(EXE)
	@touch .linking

obj/c/%.o: src/c/%.c
	@printf "${GREEN} - create object from $<${NORMAL}\n"
	$(GCC) -c $(GCCFLAGS) $< -o $@
	
obj/cu/%.o: src/cu/%.cu
	@printf "${GREEN} - create object from $<${NORMAL}\n"
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

clean:
	@printf "${GREEN} - remove the obj and dist folder${NORMAL}\n"
	rm -r -f obj dist
	
rebuild: clean | dist/$(EXE)
