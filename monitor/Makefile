# Foreground PMU sampler (see ORIGIN.md); output in monitor/build/
CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Wpedantic
CPPFLAGS := -Ithird_party/full_sched/include -Ithird_party/v3_progress
LDLIBS := -pthread
BUILD := build

all: $(BUILD)/pmu_sampler

$(BUILD)/core.o: third_party/full_sched/src/core.cpp third_party/full_sched/include/full_sched/abi.h third_party/full_sched/include/full_sched/core.h
	@mkdir -p $(BUILD)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(BUILD)/pmu_sampler: src/pmu_sampler.cpp src/progress_predictor.h third_party/v3_progress/foreground_progress.h $(BUILD)/core.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ src/pmu_sampler.cpp $(BUILD)/core.o $(LDLIBS)

$(BUILD)/progress_predictor_test: tests/progress_predictor_test.cpp src/progress_predictor.h third_party/v3_progress/foreground_progress.h
	@mkdir -p $(BUILD)
	$(CXX) $(CPPFLAGS) -Isrc $(CXXFLAGS) -o $@ $<

test: $(BUILD)/progress_predictor_test
	$(BUILD)/progress_predictor_test $(BUILD)/test_progress.page $(BUILD)/test_prediction.csv

clean:
	rm -rf $(BUILD)

.PHONY: all clean test
