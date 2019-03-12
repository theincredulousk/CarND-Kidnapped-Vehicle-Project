/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 512;
  particles.reserve(num_particles);
  
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  double x_with_noise{0.0};
  double y_with_noise = {0.0};
  double theta_with_noise = {0.0};

  for(int iParticle = 0; iParticle < num_particles; iParticle++)
  {
      x_with_noise = dist_x(_RNG);
      y_with_noise = dist_y(_RNG);
      theta_with_noise = dist_theta(_RNG);

      Particle p{iParticle, x_with_noise, y_with_noise, theta_with_noise, 1, {}, {}, {}};

      // TODO: eliminate copying of vectors in particle struct
      particles.push_back(p);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  for(auto& particle : particles)
  {
      if(yaw_rate != 0.0)
      {
          particle.x = particle.x 
            + (velocity / yaw_rate) 
            * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));

          particle.y = particle.y 
            + (velocity / yaw_rate) 
            * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));

          particle.theta = particle.theta + (yaw_rate * delta_t);
      } 
      else
      {
          particle.x = particle.x + velocity * delta_t * cos(particle.theta);
          particle.y = particle.y + velocity * delta_t * sin(particle.theta);
      }

      std::normal_distribution<double> dist_x(particle.x, std_pos[0]);
      std::normal_distribution<double> dist_y(particle.y, std_pos[1]);
      std::normal_distribution<double> dist_theta(particle.theta, std_pos[2]);

      particle.x = dist_x(_RNG);
      particle.y = dist_y(_RNG);
      particle.theta = dist_theta(_RNG);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {

  double dist_min = 1000;
  int id = 0;
  for(LandmarkObs& observed : observations)
  {
      for(LandmarkObs& predict : predicted)
      {
        double distance = dist(observed.x, observed.y ,predict.x, predict.y);
        if(distance < dist_min) 
        {
            observed.id = predict.id;
            dist_min = distance;
        }
      }
      dist_min = 1000;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  // xm,ym = map coords
  // xc, yc = car obs. coordinates
  // xp, yp = map particle coordinates
  // trans -90 deg
  // xm = xp + (cos theta * xc) - (sin theta * yc)
  // ym = yp + (sin theta * yc) - (cos theta * yc)

  for(auto& particle : particles)
  {
      double xp = particle.x;
      double yp = particle.y;
      double theta = particle.theta;
      double xc = 0.0;
      double yc = 0.0;
      double weight = 1.0;
      double xm = 0.0;
      double ym = 0.0;

      particle.associations.clear();
      particle.sense_x.clear();
      particle.sense_y.clear();

      for(const LandmarkObs& obs : observations)
      {
          xc = obs.x;
          yc = obs.y;

          xm = xp + (cos(theta) * xc) - (sin(theta) * yc);
          ym = yp + (sin(theta) * xc) + (cos(theta) * yc);

          Map::single_landmark_s closest;
          double min_dist = 1000;
          for(const Map::single_landmark_s& landmark : map_landmarks.landmark_list)
          {
              double distance = dist(xm, ym,landmark.x_f, landmark.y_f);
              if(distance < min_dist)
              {
                  min_dist = distance;
                  closest = landmark;
              }
          }

          particle.associations.push_back(closest.id_i);
          particle.sense_x.push_back(closest.x_f);
          particle.sense_y.push_back(closest.y_f);
          //std::cout << "xm, ym, closest: " << xm << "," << ym << " " << closest.x_f << ","<< closest.y_f << std::endl;
          weight *= multiv_prob(std_landmark[0], std_landmark[1], xm, ym, closest.x_f, closest.y_f);
      }

      particle.weight = weight;
      //std::cout << "weight " << weight << std::endl;
  }
}

void ParticleFilter::resample() {
  std::vector<Particle> resampled_particles;
  particles.reserve(num_particles);
  double max_weight = -10000.0;
  double sum_weight = 0.0;
  std::vector<double> weights;
  weights.reserve(num_particles);

  for(const Particle& p : particles)
  {
      if(p.weight > max_weight)
      {
        max_weight = p.weight;
      }
      sum_weight += p.weight;
      weights.push_back(p.weight);
  }

  //std::cout << "Max weight= " << max_weight << " Sum weight=" << sum_weight << std::endl;
  std::discrete_distribution<int> weight_dist(std::begin(weights), std::end(weights));
  
  int index = 0;
  for(int i = 0; i < num_particles; i++)
  {
      index = weight_dist(_RNG);
      //std::cout << "Selected index " << index << std::endl;
      resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}