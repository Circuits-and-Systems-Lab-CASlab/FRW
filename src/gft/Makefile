CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Wpedantic

TARGET := bin/ggft_homogeneous
SOURCES := src/main.cpp src/ggft_solver.cpp
HEADERS := src/ggft_solver.h

.PHONY: all clean generate test

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -Isrc $(SOURCES) -o $(TARGET)

generate: $(TARGET)
	./generate_gft.sh

test: $(TARGET)
	python3 -m pytest -q

clean:
	rm -f $(TARGET)
