/*
Points to Voxels & Voxels to Points (Modified from SparseConv)
Written by Li Jiang
All Rights Reserved 2020.
*/
#include "voxelize.h"
#include <vector>
// boziadd
#include <stdio.h>
#include <torch/torch.h>
#include <iostream>
using namespace std;
// boziadd

/* ================================== voxelize_idx
 * ================================== */
template <Int dimension>
void voxelize_idx(/* long N*4 */ at::Tensor coords,
                  /* long M*4 */ at::Tensor output_coords,
                  /* Int N */ at::Tensor input_map,
                  /* Int M*(maxActive+1) */ at::Tensor output_map,
                  Int batchSize, Int mode, at::Tensor pniv) { // boziadd
  assert(coords.ndimension() == 2);
  assert(coords.size(1) >= dimension and coords.size(1) <= dimension + 1);

  RuleBook voxelizeRuleBook;       // rule[1]: M voxels -> N points  output_map
  SparseGrids<dimension> inputSGs; // voxel_coords -> voxel_idx in M voxels
                                   // input_map: N points -> M voxels
  Int nActive = 0;
  //at::Tensor pniv;  // boziadd
/*
  Int maxActive = voxelize_inputmap<dimension>(
      inputSGs, input_map.data_ptr<Int>(), voxelizeRuleBook, nActive,
      coords.data_ptr<long>(), coords.size(0), coords.size(1), batchSize, mode, pniv.data_ptr<Int>());
*/
  returnData rtd = voxelize_inputmap<dimension>(
      inputSGs, input_map.data_ptr<Int>(), voxelizeRuleBook, nActive,
      coords.data_ptr<long>(), coords.size(0), coords.size(1), batchSize, mode, pniv.data_ptr<Int>());

  // boziadd>>> normalization
  Int maxActive = rtd.maxActive;
/*
  Int minActive = rtd.minActive;
  Int *pniv_ptr = pniv.data_ptr<Int>();
  for(Int i = 0; i < nActive; i ++){
    if(pniv_ptr[i] == 0) break;
    Int x = pniv_ptr[i];
    float fenzi = x - minActive
    float fenmu = maxActive - minActive
    if(fenzi == 0){
      pniv_ptr[i] = 0.5 * x / fenmu;
    }else{
      pniv_ptr[i] = fenzi / fenmu;
    }
  }
  cout<<"pniv_"<<pniv_<<endl;
 */ 
  // boziadd<<<
  
  output_map.resize_({nActive, maxActive + 1});
  output_map.zero_();
  
  output_coords.resize_({nActive, coords.size(1)});
  output_coords.zero_();

  Int *oM = output_map.data_ptr<Int>();
  long *oC = output_coords.data_ptr<long>();
  voxelize_outputmap<dimension>(coords.data_ptr<long>(), oC, oM,
                                &voxelizeRuleBook[1][0], nActive, maxActive);
}

/*============================================================================
* 在此处实现一个对pniv进行合并的函数，因为在推理时，如果level>2，那么就会对点进行二次体素化映射，第一次得到的pniv存的是点云个数，
而再一次的体素化映射得到的pniv存的是体素个数，voxel2是p2l_map的行，里面存的是voxel1的索引，然后voxel1的每一行里存的是点的索引，当然因为first_pniv里面存有现成的voxel1的体素内点数，
因此不需要再遍历voxel1
input: p2l_map
input: first_pniv
output: second_pniv
=========================================================================*/
// template <typename T>
// void voxelization_pniv(at::Tensor p2l_map, at::Tensor first_pniv, at::Tensor second_pniv){
//   Int *first_pniv_ptr = first_pniv.data_ptr<Int>();
//   Int *second_pniv_ptr = second_pniv.data_ptr<Int>();

//   for(Int i=0; i<p2l_map.size(0); i++){
//     for(Int j=1; j<p2l_map.size(1); j++){
//       auto voxel1_idx = p2l_map[i][j].item<Int>();
//       if(voxel1_idx != 0){
//         second_pniv_ptr[i] += first_pniv_ptr[i];
//       }
//     }
//   }
// }

template <Int dimension>
void voxelize_outputmap(long *coords, long *output_coords, Int *output_map,
                        Int *rule, Int nOutputRows, Int maxActive) {
  for (Int i = 0; i < nOutputRows; i++) {
    for (Int j = 0; j <= maxActive; j++)
      output_map[j] = rule[j];
    Int inputIdx = rule[1];
    rule += (1 + maxActive);
    output_map += (1 + maxActive);

    long *coord = coords + inputIdx * (dimension + 1);
    long *output_coord = output_coords + i * (dimension + 1);

    for (Int j = 0; j <= dimension; j++) {
      output_coord[j] = coord[j];
    }
  }
}

// mode 0=guaranteed unique 1=last item(overwrite) 2=first item(keep) 3=sum,
// 4=mean
// input: coords
// output: SGs: one map for each batch: map from voxel_coord to voxel_idx(in M
// voxels)
// output: input_map: N, N points -> M voxels
// output: rules
// output: nActive
// output: maxActive
template <Int dimension>
returnData voxelize_inputmap(SparseGrids<dimension> &SGs, Int *input_map,
                      RuleBook &rules, Int &nActive, long *coords,
                      Int nInputRows, Int nInputColumns, Int batchSize,
                      Int mode, Int *pniv) {  // boziadd
  assert(nActive == 0);
  assert(rules.size() == 0);
  assert(SGs.size() == 0);

  SGs.resize(batchSize);
  Point<dimension> p;

  std::vector<std::vector<Int>> outputRows;
  if (nInputColumns == dimension) {
    SGs.resize(1);
    auto &sg = SGs[0];
    for (Int i = 0; i < nInputRows; i++) {
      for (Int j = 0; j < dimension; j++)
        p[j] = coords[j];
      coords += dimension;
      auto iter = sg.mp.find(p);
      if (iter == sg.mp.end()) {
        sg.mp[p] = nActive++;
        outputRows.resize(nActive);
      }
      outputRows[sg.mp[p]].push_back(i);

      input_map[i] = sg.mp[p];
    }
  } else { // nInputColumns == dimension + 1 (1 in index 0 for batchidx)
    Int batchIdx;
    for (Int i = 0; i < nInputRows; i++) {
      batchIdx = coords[0];
      for (Int j = 0; j < dimension; j++)
        p[j] = coords[j + 1];
      coords += (dimension + 1);
      if (batchIdx + 1 >= (Int)SGs.size()) {
        SGs.resize(batchIdx + 1);
      }
      auto &sg = SGs[batchIdx];
      auto iter = sg.mp.find(p);
      if (iter == sg.mp.end()) {
        sg.mp[p] = nActive++;
        outputRows.resize(nActive);
      }
      outputRows[sg.mp[p]].push_back(i);

      input_map[i] = sg.mp[p];
    }
  }

  // Rulebook Format
  // rules[0][0] == mode
  // rules[0][1] == maxActive per spatial location (==1 for modes 0,1,2)
  // rules[0][2] == nInputRows
  // rules[0][3] == nOutputRows
  // rules[1]   nOutputRows x (1+maxActive)
  rules.resize(2);
  rules[0].push_back(mode);
  rules[0].push_back(1);
  rules[0].push_back(nInputRows);
  rules[0].push_back(outputRows.size());
  auto &rule = rules[1];
  if (mode == 0) {
    assert(nInputRows == (Int)outputRows.size());
    for (Int i = 0; i < nActive; i++) {
      rule.push_back(1);
      assert((Int)outputRows[i].size() == 1);
      rule.push_back(outputRows[i][0]);
    }
  }
  if (mode == 1) {
    for (Int i = 0; i < nActive; i++) {
      rule.push_back(1);
      rule.push_back(outputRows[i].front());
    }
  }
  if (mode == 2) {
    for (Int i = 0; i < nActive; i++) {
      rule.push_back(1);
      rule.push_back(outputRows[i].back());
    }
  }
  Int maxActive = 1;
  Int minActive = 1;  // boziadd
  if (mode == 3 or mode == 4) {
    Int cnt = 0;  // boziadd
    for (auto &row : outputRows){
      minActive = (Int)row.size();
      maxActive = std::max(maxActive, (Int)row.size());
      // boziadd>>>
      minActive = std::min(maxActive, (Int)row.size());
      pniv[cnt++] = (Int)row.size();  // boziadd
      // boziadd<<<
    }
      
    rules[0][1] = maxActive;
    for (auto &row : outputRows) {
      rule.push_back(row.size());
      for (auto &r : row)
        rule.push_back(r);
      rule.resize((rule.size() + maxActive) / (maxActive + 1) *
                  (maxActive + 1));
    }
  }
  // return maxActive;
  returnData rtd;
  rtd.maxActive = maxActive;
  rtd.minActive = minActive;
  return rtd;
}

/* ================================== voxelize
 * ================================== */
template <typename T>
void voxelize_fp(
    /* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
    /* cuda float M*C */ at::Tensor output_feats,
    /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int mode, Int nActive,
    Int maxActive, Int nPlane) {

  auto iF = feats.data_ptr<T>();
  auto oF = output_feats.data_ptr<T>();

  Int *rules = output_map.data_ptr<Int>();

  voxelize_fp_cuda<T>(nActive, maxActive, nPlane, iF, oF, rules, mode == 4);
}

template <typename T>
void voxelize_bp(/* cuda float M*C */ at::Tensor d_output_feats,
                 /* cuda float N*C */ at::Tensor d_feats,
                 /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int mode,
                 Int nActive, Int maxActive, Int nPlane) {
  auto d_oF = d_output_feats.data_ptr<T>();
  auto d_iF = d_feats.data_ptr<T>();

  Int *rules = output_map.data_ptr<Int>();

  voxelize_bp_cuda<T>(nActive, maxActive, nPlane, d_oF, d_iF, rules, mode == 4);
}
