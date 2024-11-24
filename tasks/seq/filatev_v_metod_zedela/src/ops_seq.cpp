// Filatev Vladislav Metod Zedela
#include "seq/filatev_v_metod_zedela/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

filatev_v_metod_zedela_seq::MetodZedela::MetodZedela(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)){
  
  this->size = taskData->inputs_count[0];
  this->delit.resize(size);

  auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
  this->matrix.insert(matrix.end(), temp, temp + size * size);

  temp = reinterpret_cast<int*>(taskData->inputs[1]);
  this->bVectrot.insert(bVectrot.end(), temp, temp + size);

}

bool filatev_v_metod_zedela_seq::MetodZedela::validation() {
  internal_order_test();

  int rank = rankMatrix(matrix, size);
  if (rank != rankRMatrix())
    return false;
  if (rank == 0 || determinant(matrix, size) == 0)
    return false;
  for (int i = 0; i < size; ++i) {
		int sum = 0;
		for (int j = 0; j < size; ++j) sum += abs(matrix[i * size + j]);
		sum -= abs(matrix[i * size + i]);
		if (sum > abs(matrix[i * size + i])){
      return false;
    }
	}
  return true;
}

bool filatev_v_metod_zedela_seq::MetodZedela::pre_processing() {
  internal_order_test();
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (i == j) { 
        delit[i] = matrix[i * size + j];
        matrix[i * size + j] = bVectrot[i]; 
      }
      else {
        matrix[i * size + j] *= -1;
      }
    }
  }
  return true;
}

bool filatev_v_metod_zedela_seq::MetodZedela::run() {
  internal_order_test();

  std::vector<double> it1(size, 0);
  std::vector<double> it2(size); //prev
  double max_z = 0;

  do{
    max_z = 0;
    swap(it1,it2);
    for (int i = 0; i < size; ++i) {
	    double sum = 0;
      for (int j = 0; j < i; ++j) {
        sum += it1[j] * matrix[i * size + j];
      }
      sum += matrix[(size + 1) * i];
      for (int j = i + 1; j < size; ++j) {
        sum += it2[j] * matrix[i * size + j];
      }
      it1[i] = double(sum) / delit[i];
    }
    for (int i = 0; i < it1.size(); ++i) {
	    max_z += abs(it1[i] - it2[i]);
    }
  } while (max_z > alfa);

  answer = it1;

  return true;
}

bool filatev_v_metod_zedela_seq::MetodZedela::post_processing() {
  internal_order_test();
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
  return true;
}

void filatev_v_metod_zedela_seq::MetodZedela::getAlfa(double _alfa){
  this->alfa = _alfa;
}

int filatev_v_metod_zedela_seq::MetodZedela::rankMatrix(std::vector<int>& matrixT, int n) {
	std::vector<double> _matrix(matrixT.size());
	std::transform(matrixT.begin(), matrixT.end(), _matrix.begin(), [](int val) {
		return static_cast<double>(val);
	});
	if (n == 0) return 0;
	int m = _matrix.size() / n;

	int rank = 0;

	for (int col = 0; col < n; col++) {
		int pivotRow = -1;
		for (int row = rank; row < m; row++) {
			if (_matrix[row * n + col] != 0) {
				pivotRow = row;
				break;
			}
		}

		if (pivotRow == -1) continue;

		if (rank != pivotRow)
			std::swap_ranges(_matrix.begin() + rank * n, _matrix.begin() + (rank + 1) * n, _matrix.begin() + pivotRow * n);

		for (int row = 0; row < m; row++) {
			if (row != rank) {
				double factor = _matrix[row * n + col] / _matrix[rank * n + col];
				for (int j = col; j < n; j++) {
					_matrix[row * n + j] -= factor * _matrix[rank * n + j];
				}
			}
		}
		rank++;
	}

	return rank;
}

double filatev_v_metod_zedela_seq::MetodZedela::determinant(std::vector<int>& _matrix, int _size) {
    if (_size == 1) {
        return _matrix[0];
    }
    if (_size == 2) {
        return _matrix[0] * _matrix[3] - _matrix[1] * _matrix[2];
    }

    double det = 0.0;

    for (int col = 0; col < _size; ++col) {
        std::vector<int> minor((_size - 1) * (_size - 1));
        for (int i = 1; i < _size; ++i) {
            for (int j = 0; j < _size; ++j) {
                if (j < col) {
                    minor[(i - 1) * (_size - 1) + j] = _matrix[i * _size + j];
                } else if (j > col) {
                    minor[(i - 1) * (_size - 1) + j - 1] = _matrix[i * _size + j];
                }
            }
        }
        det += ((col % 2 == 0) ? 1 : -1) * _matrix[col] * determinant(minor, _size - 1);
    }

    return det;
}

int filatev_v_metod_zedela_seq::MetodZedela::rankRMatrix(){
  std::vector<int> rMatrix(size * size + size);
  for (int i = 0; i < size; ++i){
    for (int j = 0; j < size; ++j){
      rMatrix[i * (size + 1) + j] = matrix[i * size + j];
    }
    rMatrix[(i + 1) * (size + 1) - 1] = bVectrot[i];
  }
  return rankMatrix(rMatrix, size + 1);
}

int filatev_v_metod_zedela_seq::TestClassForMetodZedela::generatorVector(std::vector<int>& vec){
  int sum = 0;
  for (int i = 0; i < vec.size(); ++i) {
    vec[i] = rand() % 1000 - 500;
    sum += abs(vec[i]);
  }
  return sum;
}

void filatev_v_metod_zedela_seq::TestClassForMetodZedela::generatorMatrix(std::vector<int>& matrix, int size){
  for (int i = 0; i < size; ++i) {
    std::vector<int> temp(size);
    int sum = generatorVector(temp);
    temp[i] = sum + rand() % 1000;
    matrix.insert(matrix.begin() + i * size, temp.begin(), temp.end());
  }
}

void filatev_v_metod_zedela_seq::TestClassForMetodZedela::genetatirVectorB(std::vector<int>& matrix, std::vector<int>& vecB){
  int size = vecB.size();
  this->ans.resize(size);
  generatorVector(ans);
  for (int i = 0; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < size; ++j) {
      sum += matrix[j + i * size] * ans[j];
    }
    vecB[i] = sum;
  }
}

void filatev_v_metod_zedela_seq::TestClassForMetodZedela::coutSLU(std::vector<int> matrix, std::vector<int> vecB){
  int size = vecB.size();
  std::cout << "Matrix:\n";
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      std::cout << matrix[i * size + j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "VecB:\n";
  for (int i = 0; i < size; i++) {
    std::cout << vecB[i] << " ";
  }
  std::cout << std::endl;
}

bool filatev_v_metod_zedela_seq::TestClassForMetodZedela::rightAns(std::vector<double>& answ, double alfa){
  double max_r = 0;
  for (int i = 0; i < answ.size(); ++i){
    max_r = std::max(max_r, abs(ans[i] - answ[i]));
  }
  return max_r < alfa;
}