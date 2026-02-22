#!/bin/sh

target=hw
fpga=u50

rm -rf link_${fpga}_${target}_2025
# Remove previous softFPGA files
rm -rf ../removeme/tmp0
rm ../removeme/xo/*

mkdir -p ./link_${fpga}_${target}_2025/

bash build_xo_vla.sh
cp ../removeme/xo/vla_accel.xo ./link_${fpga}_${target}_2025/

rm -rf ../removeme/tmp0
rm ../removeme/xo/*

python3 link_xo_vla_accel.py --fpga ${fpga} --target $target
# If fpga is u55c use the following line instead
if [ "$fpga" = "u50" ]; then
    cp link_${fpga}_${target}_2025/build_dir.$target.xilinx_${fpga}_gen3x16_xdma_5_202210_1/full_linked.xclbin ../host_code/
elif [ "$fpga" = "u55c" ]; then
    cp link_${fpga}_${target}_2025/build_dir.$target.xilinx_${fpga}_gen3x16_xdma_3_202210_1/full_linked.xclbin ../host_code/
elif [ "fpga" = "u280" ]; then
    cp link_${fpga}_${target}_2025/build_dir.$target.xilinx_${fpga}_gen3x16_xdma_1_202211_1/full_linked.xclbin ../host_code/
fi

# Remove junk files
rm -rf vivado_*
rm -rf v++_full_linked*
rm -rf xvlog*
rm -rf ./vivado*
rm xcd.log
