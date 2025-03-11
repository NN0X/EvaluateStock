CXX = clang++
CFLAGS = -O3 -Wall -Wextra -Wpedantic -I./include/
SRC = src
OUT = main

all:
	rm -f -r build
	mkdir build
	$(CXX) $(CFLAGS) $(SRC)/*.cpp -o build/$(OUT) 2> build/make.log
	cp -f -r data build/data
	cp LICENSE.md build/LICENSE.md
	cp NOTICE.md build/NOTICE.md
	cp README.md build/README.md
	@echo "Build complete. Executable is located at build/$(OUT)"
