// Filatev Vladislav Metod Zedela
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/filatev_v_metod_zedela/include/ops_mpi.hpp"


std::vector<std::vector<int>> getRandomMatrix(int size_n, int size_m) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int>> matrix(size_m, std::vector<int>(size_n));

  for (int i = 0; i < size_m; ++i) {
    for (int j = 0; j < size_n; ++j) {
      matrix[i][j] = gen() % 200 - 100;
    }
  }
  return matrix;
}


TEST(filatev_v_metod_zedela_mpi, test1) {
  boost::mpi::communicator world;
  std::vector<int> out;
  std::vector<std::vector<int>> in;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count = 10;
    in = std::vector<std::vector<int>>(count, std::vector<int>(count, 1));
    out = std::vector<int>(1, 0);
    for (int i = 0; i < count; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  filatev_v_metod_zedela_mpi::SumMatrixParallel sumMatrixparallel(taskDataPar, world);
  ASSERT_EQ(sumMatrixparallel.validation(), true);
  sumMatrixparallel.pre_processing();
  sumMatrixparallel.run();
  sumMatrixparallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(100, out[0]);
  }
}

TEST(filatev_v_metod_zedela_mpi, test2) {
  boost::mpi::communicator world;
  const int count = 10;
  std::vector<int> out;
  std::vector<std::vector<int>> in;
  std::vector<std::vector<int>> refIn;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = getRandomMatrix(count, count);
    refIn = in;
    out = std::vector<int>(1, 0);
    for (int i = 0; i < count; i++) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in[i].data()));
    }
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->inputs_count.emplace_back(count);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  filatev_v_metod_zedela_mpi::SumMatrixParallel sumMatrixParallel(taskDataPar, world);
  ASSERT_EQ(sumMatrixParallel.validation(), true);
  sumMatrixParallel.pre_processing();
  sumMatrixParallel.run();
  sumMatrixParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> refOut;
    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> TaskDataSeq = std::make_shared<ppc::core::TaskData>();
    refOut = std::vector<int>(1, 0);
    for (int i = 0; i < count; i++) {
      TaskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(refIn[i].data()));
    }
    TaskDataSeq->inputs_count.emplace_back(count);
    TaskDataSeq->inputs_count.emplace_back(count);
    TaskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(refOut.data()));
    TaskDataSeq->outputs_count.emplace_back(1);

    filatev_v_metod_zedela_mpi::SumMatrixSeq sumMatriSeq(TaskDataSeq);
    ASSERT_EQ(sumMatriSeq.validation(), true);
    sumMatriSeq.pre_processing();
    sumMatriSeq.run();
    sumMatriSeq.post_processing();

    ASSERT_EQ(out[0], refOut[0]);
  }
}
