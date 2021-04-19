#pragma once

#include "IPlug_include_in_plug_hdr.h"

#include "SpeexResampler.h"
#include <vector>
#include <memory>
#include <complex>

const int kNumPresets = 1;

enum EParams
{
  kGain = 0,
  kInternalSampleRate,
  kInternalLPOrder,
  kNumFormantsOut,
  kNumParams
};

using namespace iplug;
using namespace igraphics;

struct LpcWork {
  std::vector<double> b1;
  std::vector<double> b2;
  std::vector<double> aa;
  std::vector<double> lpc;
};

struct RootsWork {
  std::vector<std::complex<double>> P;
  std::vector<std::complex<double>> P2;
  std::vector<std::complex<double>> Q;
  std::vector<std::complex<double>> R;
};

class InFormantMidi final : public Plugin
{
public:
  InFormantMidi(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
  void OnReset() override;
  void OnParamChange(int paramIdx) override;

private:
  speexport::SpeexResampler mResampler;
  double mInternalFs;
  std::vector<float> mMixedBlockIn;
  std::vector<float> mMixedBlockOut;
  std::vector<float> mInput;
  std::vector<float> mWindow;
  std::vector<double> mWork;
  LpcWork mLpcWork;
  RootsWork mRootsWork;
  std::vector<double> mAllFormants;
  std::vector<double> mFormants;
#endif
};

#if IPLUG_DSP
namespace Math {
  std::vector<float> makeGaussianWindow(int nSamples);

  bool calculateLpc(const std::vector<double>& input, LpcWork& work);

  template<int nd>
  std::array<std::complex<double>, nd + 1> calculatePolynomialWithDerivatives(const std::vector<std::complex<double>>& P, const std::complex<double>& x)
  {
    const int degree = P.size() - 1;
    std::array<std::complex<double>, nd + 1> y;

    std::fill(y.begin(), y.end(), 0.0);
    y[0] = P[degree];

    for (int i = degree - 1; i >= 0; --i) {
      const int n = nd < degree - i ? nd : degree - i;
      for (int j = n; j >= 1; --j) {
        y[j] = y[j] * x + y[j - 1];
      }
      y[0] = y[0] * x + P[i];
    }

    double fact = 1.0;
    for (int j = 2; j <= nd; ++j) {
      fact *= j;
      y[j] *= fact;
    }
    return y;
  }

  std::complex<double> calculateOneRoot(const std::vector<std::complex<double>>& P, std::complex<double> xk, double accuracy);
  
  void deflatePolynomial(const std::vector<std::complex<double>>& P, const std::complex<double>& r, std::vector<std::complex<double>>& Q);

  void calculateRoots(RootsWork& work);

  void processBlock(
    const std::vector<float>& input,
    const std::vector<float>& window,
    double Fs,
    std::vector<double>& work,
    LpcWork& lpcWork,
    RootsWork& rootsWork,
    std::vector<double>& allFormants,
    std::vector<double>& formants);
}
#endif
