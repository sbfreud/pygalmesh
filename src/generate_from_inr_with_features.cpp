#define CGAL_MESH_3_VERBOSE 1

#include "generate_from_inr_with_features.hpp"

#include <cassert>

#include <vector>
#include <iostream>
 
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
 
#include <CGAL/make_mesh_3.h>
#include <CGAL/Image_3.h>
 
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_domain_with_polyline_features_3.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/Mesh_3/Detect_features_in_image.h>
 

namespace pygalmesh {

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Labeled_mesh_domain_3<K> Image_domain;
typedef CGAL::Mesh_domain_with_polyline_features_3<Image_domain> Mesh_domain;
 

#ifdef CGAL_CONCURRENT_MESH_3
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif
 
// Triangulation
typedef CGAL::Mesh_triangulation_3<Mesh_domain,CGAL::Default,Concurrency_tag>::type Tr;
 
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;
 
// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;
// To avoid verbose function and named parameters call
// namespace params = CGAL::parameters;
void 
generate_from_inr_with_features(
    const std::string & inr_filename,
    const std::string & outfile,
    const bool lloyd,
    const bool odt,
    const bool perturb,
    const bool exude,
    const double max_edge_size_at_feature_edges,
    const double min_facet_angle,
    const double max_radius_surface_delaunay_ball,
    const double max_facet_distance,
    const double max_circumradius_edge_ratio,
    const double max_cell_circumradius,
    const double exude_time_limit,
    const double exude_sliver_bound,
    const bool verbose
    )
{

  CGAL::Image_3 image;
  const bool success = image.read(inr_filename.c_str());
  if (!success) {
    throw "Could not read image file";
  }


  Mesh_domain cgal_domain = Mesh_domain::create_labeled_image_mesh_domain(image,
                                                                          CGAL::parameters::features_detector = CGAL::Mesh_3::Detect_features_in_image());
 

  Mesh_criteria criteria(
      CGAL::parameters::edge_size = max_edge_size_at_feature_edges,
      CGAL::parameters::facet_angle = min_facet_angle,
      CGAL::parameters::facet_size = max_radius_surface_delaunay_ball,
      CGAL::parameters::facet_distance = max_facet_distance,
      // CGAL::parameters::facet_topology = CGAL::FACET_VERTICES_ON_SAME_SURFACE_PATCH,
      CGAL::parameters::cell_radius_edge_ratio = max_circumradius_edge_ratio,
      CGAL::parameters::cell_size = max_cell_circumradius
      );

  // Mesh generation
  if (!verbose) {
    // suppress output
    std::cerr.setstate(std::ios_base::failbit);
  }
  C3t3 c3t3 = CGAL::make_mesh_3<C3t3>(
      cgal_domain,
      criteria,
      lloyd ? CGAL::parameters::lloyd() : CGAL::parameters::no_lloyd(),
      odt ? CGAL::parameters::odt() : CGAL::parameters::no_odt(),
      perturb ? CGAL::parameters::perturb(
          CGAL::parameters::time_limit = exude_time_limit,
          CGAL::parameters::sliver_bound = exude_sliver_bound
        ) : 
        CGAL::parameters::no_perturb(),
      exude ?
        CGAL::parameters::exude(
          CGAL::parameters::time_limit = exude_time_limit,
          CGAL::parameters::sliver_bound = exude_sliver_bound
        ) :
        CGAL::parameters::no_exude()
      );
  if (!verbose) {
    std::cerr.clear();
  }

  // Output
  std::ofstream medit_file(outfile);
  c3t3.output_to_medit(medit_file);
  medit_file.close();
  return;
}


} // namespace pygalmesh
