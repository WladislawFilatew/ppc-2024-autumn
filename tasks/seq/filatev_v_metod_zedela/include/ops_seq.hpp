// Filatev Vladislav Metod Zedela
#pragma once

#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include "core/task/include/task.hpp"

namespace filatev_v_metod_zedela_seq {

class MetodZedela : public ppc::core::Task {
public:
  explicit MetodZedela(std::shared_ptr<ppc::core::TaskData> taskData_);
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  void getAlfa(double alfa);
  int rankMatrix(std::vector<int>& matrixT, int n);
  int rankRMatrix();
  double determinant();

private:
  int size;
  double alfa;
  std::vector<int> matrix;
  std::vector<int> bVectrot;
  std::vector<double> answer;
  std::vector<int> delit;
};

class TestClassForMetodZedela {
public:
  int generatorVector(std::vector<int>& vec);
  void generatorMatrix(std::vector<int>& matrix, int size);
  void genetatirVectorB(std::vector<int>& matrix, std::vector<int>& vecB);
  void coutSLU(std::vector<int> matrix, std::vector<int> vecB);
  bool rightAns(std::vector<double>& ans, double alfa);
private:
  std::vector<int> ans;
};

}  // namespace filatev_v_metod_zedela_seq

