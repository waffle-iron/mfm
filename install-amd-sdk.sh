bash amd_sdk.sh;
cd /tmp
tar -xjf AMD-SDK.tar.bz2;
AMDAPPSDK=${HOME}/AMDAPPSDK;
export OPENCL_VENDOR_PATH=${AMDAPPSDK}/etc/OpenCL/vendors;
mkdir -p ${OPENCL_VENDOR_PATH};
sh AMD-APP-SDK*.sh --tar -xf -C ${AMDAPPSDK};
echo libamdocl64.so > ${OPENCL_VENDOR_PATH}/amdocl64.icd;
export LD_LIBRARY_PATH=${AMDAPPSDK}/lib/x86_64:${LD_LIBRARY_PATH};
chmod +x ${AMDAPPSDK}/bin/x86_64/clinfo;
${AMDAPPSDK}/bin/x86_64/clinfo;
ln -s /root/AMDAPPSDK/lib/x86_64/libOpenCL.so /usr/lib/libOpenCL.so
cd -
