CC=gcc
CUARCH=sm_20
CCDEBUG=
CUDEBUG=
OPT=O3
CU=/usr/local/cuda-6.5/bin/nvcc -arch=$(CUARCH)
CCFLAGS=-c -Wall -$(OPT)
CUFLAGS=-dc -$(OPT)

CCSRC=
CUSRC=oldkernel.cu
CCSRCDIR=src/cc
CUSRCDIR=src/cu

CCOBJDIR=obj_cc
CUOBJDIR=obj_cu

CCOBJECTS=$(addprefix $(CCOBJDIR)/,$(CCSRC:.cpp=.o))
CUOBJECTS=$(addprefix $(CUOBJDIR)/,$(CUSRC:.cu=.o))

EXE=GPUPhysics-v2
EXEDIR=bin
all: $(EXE)

nolink: $(CCOBJECTS) $(CUOBJECTS)

$(EXE): $(CCOBJECTS) $(CUOBJECTS)
	$(CU) $^ -o $(EXEDIR)/$(EXE)					# link final executable
	
-include $(CCOBJECTS:.o=_cc.d)						# import generated dependencies
-include $(CUOBJECTS:.o=_cu.d)
	
$(CCOBJDIR)/%.o: $(CCSRCDIR)/%.cpp | $(CCOBJDIR)
	$(CC) $(CCFLAGS) $(CCDEBUG) $< -o $@					# compile C++ source-files
	@$(CC) -MM $< > $(CCOBJDIR)/$*_cc.d				# generate dependencies
	@sed -i '1s/^/$(CCOBJDIR)\//' $(CCOBJDIR)/$*_cc.d		# prepend object-dir to the target
	
$(CUOBJDIR)/%.o: $(CUSRCDIR)/%.cu | $(CUOBJDIR)
	$(CU) $(CUFLAGS) $(CUDEBUG) $< -o $@					# compile CUDA-C++ source-files
	@$(CC) -x c++ -MM $< > $(CUOBJDIR)/$*_cu.d			# generate dependencies
	@sed -i '1s/^/$(CUOBJDIR)\//' $(CUOBJDIR)/$*_cu.d		# prepend object-dir to the target
	
$(CCOBJDIR):
	@mkdir -p $@
	
$(CUOBJDIR):
	@mkdir -p $@
	
clean:
	rm -f $(CCOBJDIR)/*.o $(CUOBJDIR)/*.o $(CCOBJDIR)/*.d $(CUOBJDIR)/*.d
