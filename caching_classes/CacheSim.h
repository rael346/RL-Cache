#pragma once

#include <Python.h>
#include <boost/python.hpp>
#include <map>
#include <random>
#include <unordered_set>
#include <vector>

using std::map;
using std::multimap;
using std::unordered_set;
using std::vector;

typedef multimap<double, uint64_t>::iterator mm_iterator;

namespace p = boost::python;

class CacheSim {
public:
  CacheSim(uint64_t cache_size);

  virtual void reset();

  virtual double hit_rate();

  virtual double byte_hit_rate();

  virtual uint64_t free_space();

  virtual bool decide(p::dict request, double eviction_rating,
                      int admission_decision);

  bool prediction_updated_eviction;
  bool prediction_updated_admission;

  uint64_t refresh_period;

  virtual p::dict get_ratings();
  virtual void set_ratings(p::dict &_ratings);

  p::dict get_sizes();
  void set_sizes(p::dict &_sizes);

  uint64_t get_used_space();
  void set_used_space(uint64_t _used_space);

  uint64_t get_cache_size();
  void set_cache_size(uint64_t _cache_size);

  double get_L();
  void set_L(double _L);

  uint64_t get_misses();
  void set_misses(uint64_t _misses);

  uint64_t get_hits();
  void set_hits(uint64_t _hits);

  uint64_t get_byte_misses();
  void set_byte_misses(uint64_t _byte_misses);

  uint64_t get_byte_hits();
  void set_byte_hits(uint64_t _byte_hits);

  double get_total_rating();
  void set_total_rating(double _total_rating);

  bool get_ml_eviction();
  void set_ml_eviction(double _is_ml_eviction);

protected:
  virtual void produce_new_cache_state(p::dict &request, double eviction_rating,
                                       int admission_decision) = 0;

  map<uint64_t, mm_iterator> cache;
  multimap<double, uint64_t> ratings;
  map<uint64_t, uint64_t> sizes;

  uint64_t used_space;

  double L;

  uint64_t hits;
  uint64_t misses;

  double byte_hits;
  double byte_misses;

  uint64_t cache_size;

  double total_rating;

  bool is_ml_eviction;

  std::mt19937 generator;
  std::uniform_real_distribution<> distr;
};
