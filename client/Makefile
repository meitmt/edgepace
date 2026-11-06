# edgepace_client (see edgepace_client.cpp); output in client/build/
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS := -I../llama.cpp/ggml/src
BUILD := build

all: $(BUILD)/edgepace_client

$(BUILD)/edgepace_client: edgepace_client.cpp edgepace_request_page.h ../llama.cpp/ggml/src/edgepace/abi.h
	@mkdir -p $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ edgepace_client.cpp -pthread

clean:
	rm -rf $(BUILD)

.PHONY: all clean
