#include "LRUSim.h"
#include <sstream>

LRUSimulator::LRUSimulator(uint64_t _cache_size) : CacheSim(_cache_size) {}

std::string LRUSimulator::toString() {
  std::ostringstream oss;
  oss << "AdaptSizeSimulator(cache_size=" << cache_size << ")";
  return oss.str();
}

void LRUSimulator::produce_new_cache_state(p::dict &request,
                                           double eviction_rating,
                                           int admission_decision) {
  uint64_t size = p::extract<uint64_t>(request.get("size"));

  if (!admission_decision) {
    return;
  }

  prediction_updated_eviction = true;

  while (used_space + size > cache_size) {
    auto min_elem = ratings.begin();

    used_space -= sizes[min_elem->second];

    cache.erase(min_elem->second);
    sizes.erase(min_elem->second);
    ratings.erase(min_elem);
  }

  double prediction = eviction_rating;
  uint64_t id = p::extract<uint64_t>(request.get("id"));

  double rating = prediction;
  total_rating += rating;
  cache[id] = ratings.emplace(rating, id);
  sizes.insert(pair<uint64_t, uint64_t>(id, size));
  used_space += size;

  return;
}
