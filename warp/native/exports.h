namespace wp {

extern "C" {

WP_API void builtin_min_float16_float16(float16 a, float16 b, float16* ret);
WP_API void builtin_min_float32_float32(float32 a, float32 b, float32* ret);
WP_API void builtin_min_float64_float64(float64 a, float64 b, float64* ret);
WP_API void builtin_min_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_min_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_min_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_min_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_min_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_min_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_min_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_min_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_min_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret);
WP_API void builtin_min_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret);
WP_API void builtin_min_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret);
WP_API void builtin_min_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_min_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret);
WP_API void builtin_min_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret);
WP_API void builtin_min_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret);
WP_API void builtin_min_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_min_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret);
WP_API void builtin_min_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret);
WP_API void builtin_min_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret);
WP_API void builtin_min_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_min_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret);
WP_API void builtin_min_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret);
WP_API void builtin_min_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret);
WP_API void builtin_min_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret);
WP_API void builtin_min_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret);
WP_API void builtin_min_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret);
WP_API void builtin_min_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret);
WP_API void builtin_min_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret);
WP_API void builtin_min_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret);
WP_API void builtin_min_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret);
WP_API void builtin_min_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret);
WP_API void builtin_min_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret);
WP_API void builtin_min_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret);
WP_API void builtin_min_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret);
WP_API void builtin_min_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret);
WP_API void builtin_min_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret);
WP_API void builtin_min_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret);
WP_API void builtin_min_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret);
WP_API void builtin_min_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret);
WP_API void builtin_min_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret);
WP_API void builtin_min_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret);
WP_API void builtin_min_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret);
WP_API void builtin_min_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret);
WP_API void builtin_min_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret);
WP_API void builtin_min_vec2h(vec2h& a, float16* ret);
WP_API void builtin_min_vec3h(vec3h& a, float16* ret);
WP_API void builtin_min_vec4h(vec4h& a, float16* ret);
WP_API void builtin_min_spatial_vectorh(spatial_vectorh& a, float16* ret);
WP_API void builtin_min_vec2f(vec2f& a, float32* ret);
WP_API void builtin_min_vec3f(vec3f& a, float32* ret);
WP_API void builtin_min_vec4f(vec4f& a, float32* ret);
WP_API void builtin_min_spatial_vectorf(spatial_vectorf& a, float32* ret);
WP_API void builtin_min_vec2d(vec2d& a, float64* ret);
WP_API void builtin_min_vec3d(vec3d& a, float64* ret);
WP_API void builtin_min_vec4d(vec4d& a, float64* ret);
WP_API void builtin_min_spatial_vectord(spatial_vectord& a, float64* ret);
WP_API void builtin_min_vec2s(vec2s& a, int16* ret);
WP_API void builtin_min_vec3s(vec3s& a, int16* ret);
WP_API void builtin_min_vec4s(vec4s& a, int16* ret);
WP_API void builtin_min_vec2i(vec2i& a, int32* ret);
WP_API void builtin_min_vec3i(vec3i& a, int32* ret);
WP_API void builtin_min_vec4i(vec4i& a, int32* ret);
WP_API void builtin_min_vec2l(vec2l& a, int64* ret);
WP_API void builtin_min_vec3l(vec3l& a, int64* ret);
WP_API void builtin_min_vec4l(vec4l& a, int64* ret);
WP_API void builtin_min_vec2b(vec2b& a, int8* ret);
WP_API void builtin_min_vec3b(vec3b& a, int8* ret);
WP_API void builtin_min_vec4b(vec4b& a, int8* ret);
WP_API void builtin_min_vec2us(vec2us& a, uint16* ret);
WP_API void builtin_min_vec3us(vec3us& a, uint16* ret);
WP_API void builtin_min_vec4us(vec4us& a, uint16* ret);
WP_API void builtin_min_vec2ui(vec2ui& a, uint32* ret);
WP_API void builtin_min_vec3ui(vec3ui& a, uint32* ret);
WP_API void builtin_min_vec4ui(vec4ui& a, uint32* ret);
WP_API void builtin_min_vec2ul(vec2ul& a, uint64* ret);
WP_API void builtin_min_vec3ul(vec3ul& a, uint64* ret);
WP_API void builtin_min_vec4ul(vec4ul& a, uint64* ret);
WP_API void builtin_min_vec2ub(vec2ub& a, uint8* ret);
WP_API void builtin_min_vec3ub(vec3ub& a, uint8* ret);
WP_API void builtin_min_vec4ub(vec4ub& a, uint8* ret);
WP_API void builtin_max_float16_float16(float16 a, float16 b, float16* ret);
WP_API void builtin_max_float32_float32(float32 a, float32 b, float32* ret);
WP_API void builtin_max_float64_float64(float64 a, float64 b, float64* ret);
WP_API void builtin_max_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_max_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_max_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_max_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_max_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_max_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_max_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_max_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_max_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret);
WP_API void builtin_max_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret);
WP_API void builtin_max_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret);
WP_API void builtin_max_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_max_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret);
WP_API void builtin_max_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret);
WP_API void builtin_max_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret);
WP_API void builtin_max_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_max_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret);
WP_API void builtin_max_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret);
WP_API void builtin_max_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret);
WP_API void builtin_max_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_max_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret);
WP_API void builtin_max_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret);
WP_API void builtin_max_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret);
WP_API void builtin_max_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret);
WP_API void builtin_max_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret);
WP_API void builtin_max_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret);
WP_API void builtin_max_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret);
WP_API void builtin_max_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret);
WP_API void builtin_max_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret);
WP_API void builtin_max_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret);
WP_API void builtin_max_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret);
WP_API void builtin_max_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret);
WP_API void builtin_max_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret);
WP_API void builtin_max_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret);
WP_API void builtin_max_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret);
WP_API void builtin_max_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret);
WP_API void builtin_max_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret);
WP_API void builtin_max_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret);
WP_API void builtin_max_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret);
WP_API void builtin_max_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret);
WP_API void builtin_max_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret);
WP_API void builtin_max_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret);
WP_API void builtin_max_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret);
WP_API void builtin_max_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret);
WP_API void builtin_max_vec2h(vec2h& a, float16* ret);
WP_API void builtin_max_vec3h(vec3h& a, float16* ret);
WP_API void builtin_max_vec4h(vec4h& a, float16* ret);
WP_API void builtin_max_spatial_vectorh(spatial_vectorh& a, float16* ret);
WP_API void builtin_max_vec2f(vec2f& a, float32* ret);
WP_API void builtin_max_vec3f(vec3f& a, float32* ret);
WP_API void builtin_max_vec4f(vec4f& a, float32* ret);
WP_API void builtin_max_spatial_vectorf(spatial_vectorf& a, float32* ret);
WP_API void builtin_max_vec2d(vec2d& a, float64* ret);
WP_API void builtin_max_vec3d(vec3d& a, float64* ret);
WP_API void builtin_max_vec4d(vec4d& a, float64* ret);
WP_API void builtin_max_spatial_vectord(spatial_vectord& a, float64* ret);
WP_API void builtin_max_vec2s(vec2s& a, int16* ret);
WP_API void builtin_max_vec3s(vec3s& a, int16* ret);
WP_API void builtin_max_vec4s(vec4s& a, int16* ret);
WP_API void builtin_max_vec2i(vec2i& a, int32* ret);
WP_API void builtin_max_vec3i(vec3i& a, int32* ret);
WP_API void builtin_max_vec4i(vec4i& a, int32* ret);
WP_API void builtin_max_vec2l(vec2l& a, int64* ret);
WP_API void builtin_max_vec3l(vec3l& a, int64* ret);
WP_API void builtin_max_vec4l(vec4l& a, int64* ret);
WP_API void builtin_max_vec2b(vec2b& a, int8* ret);
WP_API void builtin_max_vec3b(vec3b& a, int8* ret);
WP_API void builtin_max_vec4b(vec4b& a, int8* ret);
WP_API void builtin_max_vec2us(vec2us& a, uint16* ret);
WP_API void builtin_max_vec3us(vec3us& a, uint16* ret);
WP_API void builtin_max_vec4us(vec4us& a, uint16* ret);
WP_API void builtin_max_vec2ui(vec2ui& a, uint32* ret);
WP_API void builtin_max_vec3ui(vec3ui& a, uint32* ret);
WP_API void builtin_max_vec4ui(vec4ui& a, uint32* ret);
WP_API void builtin_max_vec2ul(vec2ul& a, uint64* ret);
WP_API void builtin_max_vec3ul(vec3ul& a, uint64* ret);
WP_API void builtin_max_vec4ul(vec4ul& a, uint64* ret);
WP_API void builtin_max_vec2ub(vec2ub& a, uint8* ret);
WP_API void builtin_max_vec3ub(vec3ub& a, uint8* ret);
WP_API void builtin_max_vec4ub(vec4ub& a, uint8* ret);
WP_API void builtin_clamp_float16_float16_float16(float16 x, float16 low, float16 high, float16* ret);
WP_API void builtin_clamp_float32_float32_float32(float32 x, float32 low, float32 high, float32* ret);
WP_API void builtin_clamp_float64_float64_float64(float64 x, float64 low, float64 high, float64* ret);
WP_API void builtin_clamp_int16_int16_int16(int16 x, int16 low, int16 high, int16* ret);
WP_API void builtin_clamp_int32_int32_int32(int32 x, int32 low, int32 high, int32* ret);
WP_API void builtin_clamp_int64_int64_int64(int64 x, int64 low, int64 high, int64* ret);
WP_API void builtin_clamp_int8_int8_int8(int8 x, int8 low, int8 high, int8* ret);
WP_API void builtin_clamp_uint16_uint16_uint16(uint16 x, uint16 low, uint16 high, uint16* ret);
WP_API void builtin_clamp_uint32_uint32_uint32(uint32 x, uint32 low, uint32 high, uint32* ret);
WP_API void builtin_clamp_uint64_uint64_uint64(uint64 x, uint64 low, uint64 high, uint64* ret);
WP_API void builtin_clamp_uint8_uint8_uint8(uint8 x, uint8 low, uint8 high, uint8* ret);
WP_API void builtin_abs_float16(float16 x, float16* ret);
WP_API void builtin_abs_float32(float32 x, float32* ret);
WP_API void builtin_abs_float64(float64 x, float64* ret);
WP_API void builtin_abs_int16(int16 x, int16* ret);
WP_API void builtin_abs_int32(int32 x, int32* ret);
WP_API void builtin_abs_int64(int64 x, int64* ret);
WP_API void builtin_abs_int8(int8 x, int8* ret);
WP_API void builtin_abs_uint16(uint16 x, uint16* ret);
WP_API void builtin_abs_uint32(uint32 x, uint32* ret);
WP_API void builtin_abs_uint64(uint64 x, uint64* ret);
WP_API void builtin_abs_uint8(uint8 x, uint8* ret);
WP_API void builtin_abs_vec2h(vec2h& x, vec2h* ret);
WP_API void builtin_abs_vec3h(vec3h& x, vec3h* ret);
WP_API void builtin_abs_vec4h(vec4h& x, vec4h* ret);
WP_API void builtin_abs_spatial_vectorh(spatial_vectorh& x, spatial_vectorh* ret);
WP_API void builtin_abs_vec2f(vec2f& x, vec2f* ret);
WP_API void builtin_abs_vec3f(vec3f& x, vec3f* ret);
WP_API void builtin_abs_vec4f(vec4f& x, vec4f* ret);
WP_API void builtin_abs_spatial_vectorf(spatial_vectorf& x, spatial_vectorf* ret);
WP_API void builtin_abs_vec2d(vec2d& x, vec2d* ret);
WP_API void builtin_abs_vec3d(vec3d& x, vec3d* ret);
WP_API void builtin_abs_vec4d(vec4d& x, vec4d* ret);
WP_API void builtin_abs_spatial_vectord(spatial_vectord& x, spatial_vectord* ret);
WP_API void builtin_abs_vec2s(vec2s& x, vec2s* ret);
WP_API void builtin_abs_vec3s(vec3s& x, vec3s* ret);
WP_API void builtin_abs_vec4s(vec4s& x, vec4s* ret);
WP_API void builtin_abs_vec2i(vec2i& x, vec2i* ret);
WP_API void builtin_abs_vec3i(vec3i& x, vec3i* ret);
WP_API void builtin_abs_vec4i(vec4i& x, vec4i* ret);
WP_API void builtin_abs_vec2l(vec2l& x, vec2l* ret);
WP_API void builtin_abs_vec3l(vec3l& x, vec3l* ret);
WP_API void builtin_abs_vec4l(vec4l& x, vec4l* ret);
WP_API void builtin_abs_vec2b(vec2b& x, vec2b* ret);
WP_API void builtin_abs_vec3b(vec3b& x, vec3b* ret);
WP_API void builtin_abs_vec4b(vec4b& x, vec4b* ret);
WP_API void builtin_abs_vec2us(vec2us& x, vec2us* ret);
WP_API void builtin_abs_vec3us(vec3us& x, vec3us* ret);
WP_API void builtin_abs_vec4us(vec4us& x, vec4us* ret);
WP_API void builtin_abs_vec2ui(vec2ui& x, vec2ui* ret);
WP_API void builtin_abs_vec3ui(vec3ui& x, vec3ui* ret);
WP_API void builtin_abs_vec4ui(vec4ui& x, vec4ui* ret);
WP_API void builtin_abs_vec2ul(vec2ul& x, vec2ul* ret);
WP_API void builtin_abs_vec3ul(vec3ul& x, vec3ul* ret);
WP_API void builtin_abs_vec4ul(vec4ul& x, vec4ul* ret);
WP_API void builtin_abs_vec2ub(vec2ub& x, vec2ub* ret);
WP_API void builtin_abs_vec3ub(vec3ub& x, vec3ub* ret);
WP_API void builtin_abs_vec4ub(vec4ub& x, vec4ub* ret);
WP_API void builtin_sign_float16(float16 x, float16* ret);
WP_API void builtin_sign_float32(float32 x, float32* ret);
WP_API void builtin_sign_float64(float64 x, float64* ret);
WP_API void builtin_sign_int16(int16 x, int16* ret);
WP_API void builtin_sign_int32(int32 x, int32* ret);
WP_API void builtin_sign_int64(int64 x, int64* ret);
WP_API void builtin_sign_int8(int8 x, int8* ret);
WP_API void builtin_sign_uint16(uint16 x, uint16* ret);
WP_API void builtin_sign_uint32(uint32 x, uint32* ret);
WP_API void builtin_sign_uint64(uint64 x, uint64* ret);
WP_API void builtin_sign_uint8(uint8 x, uint8* ret);
WP_API void builtin_sign_vec2h(vec2h& x, vec2h* ret);
WP_API void builtin_sign_vec3h(vec3h& x, vec3h* ret);
WP_API void builtin_sign_vec4h(vec4h& x, vec4h* ret);
WP_API void builtin_sign_spatial_vectorh(spatial_vectorh& x, spatial_vectorh* ret);
WP_API void builtin_sign_vec2f(vec2f& x, vec2f* ret);
WP_API void builtin_sign_vec3f(vec3f& x, vec3f* ret);
WP_API void builtin_sign_vec4f(vec4f& x, vec4f* ret);
WP_API void builtin_sign_spatial_vectorf(spatial_vectorf& x, spatial_vectorf* ret);
WP_API void builtin_sign_vec2d(vec2d& x, vec2d* ret);
WP_API void builtin_sign_vec3d(vec3d& x, vec3d* ret);
WP_API void builtin_sign_vec4d(vec4d& x, vec4d* ret);
WP_API void builtin_sign_spatial_vectord(spatial_vectord& x, spatial_vectord* ret);
WP_API void builtin_sign_vec2s(vec2s& x, vec2s* ret);
WP_API void builtin_sign_vec3s(vec3s& x, vec3s* ret);
WP_API void builtin_sign_vec4s(vec4s& x, vec4s* ret);
WP_API void builtin_sign_vec2i(vec2i& x, vec2i* ret);
WP_API void builtin_sign_vec3i(vec3i& x, vec3i* ret);
WP_API void builtin_sign_vec4i(vec4i& x, vec4i* ret);
WP_API void builtin_sign_vec2l(vec2l& x, vec2l* ret);
WP_API void builtin_sign_vec3l(vec3l& x, vec3l* ret);
WP_API void builtin_sign_vec4l(vec4l& x, vec4l* ret);
WP_API void builtin_sign_vec2b(vec2b& x, vec2b* ret);
WP_API void builtin_sign_vec3b(vec3b& x, vec3b* ret);
WP_API void builtin_sign_vec4b(vec4b& x, vec4b* ret);
WP_API void builtin_sign_vec2us(vec2us& x, vec2us* ret);
WP_API void builtin_sign_vec3us(vec3us& x, vec3us* ret);
WP_API void builtin_sign_vec4us(vec4us& x, vec4us* ret);
WP_API void builtin_sign_vec2ui(vec2ui& x, vec2ui* ret);
WP_API void builtin_sign_vec3ui(vec3ui& x, vec3ui* ret);
WP_API void builtin_sign_vec4ui(vec4ui& x, vec4ui* ret);
WP_API void builtin_sign_vec2ul(vec2ul& x, vec2ul* ret);
WP_API void builtin_sign_vec3ul(vec3ul& x, vec3ul* ret);
WP_API void builtin_sign_vec4ul(vec4ul& x, vec4ul* ret);
WP_API void builtin_sign_vec2ub(vec2ub& x, vec2ub* ret);
WP_API void builtin_sign_vec3ub(vec3ub& x, vec3ub* ret);
WP_API void builtin_sign_vec4ub(vec4ub& x, vec4ub* ret);
WP_API void builtin_step_float16(float16 x, float16* ret);
WP_API void builtin_step_float32(float32 x, float32* ret);
WP_API void builtin_step_float64(float64 x, float64* ret);
WP_API void builtin_step_int16(int16 x, int16* ret);
WP_API void builtin_step_int32(int32 x, int32* ret);
WP_API void builtin_step_int64(int64 x, int64* ret);
WP_API void builtin_step_int8(int8 x, int8* ret);
WP_API void builtin_step_uint16(uint16 x, uint16* ret);
WP_API void builtin_step_uint32(uint32 x, uint32* ret);
WP_API void builtin_step_uint64(uint64 x, uint64* ret);
WP_API void builtin_step_uint8(uint8 x, uint8* ret);
WP_API void builtin_nonzero_float16(float16 x, float16* ret);
WP_API void builtin_nonzero_float32(float32 x, float32* ret);
WP_API void builtin_nonzero_float64(float64 x, float64* ret);
WP_API void builtin_nonzero_int16(int16 x, int16* ret);
WP_API void builtin_nonzero_int32(int32 x, int32* ret);
WP_API void builtin_nonzero_int64(int64 x, int64* ret);
WP_API void builtin_nonzero_int8(int8 x, int8* ret);
WP_API void builtin_nonzero_uint16(uint16 x, uint16* ret);
WP_API void builtin_nonzero_uint32(uint32 x, uint32* ret);
WP_API void builtin_nonzero_uint64(uint64 x, uint64* ret);
WP_API void builtin_nonzero_uint8(uint8 x, uint8* ret);
WP_API void builtin_sin_float16(float16 x, float16* ret);
WP_API void builtin_sin_float32(float32 x, float32* ret);
WP_API void builtin_sin_float64(float64 x, float64* ret);
WP_API void builtin_cos_float16(float16 x, float16* ret);
WP_API void builtin_cos_float32(float32 x, float32* ret);
WP_API void builtin_cos_float64(float64 x, float64* ret);
WP_API void builtin_acos_float16(float16 x, float16* ret);
WP_API void builtin_acos_float32(float32 x, float32* ret);
WP_API void builtin_acos_float64(float64 x, float64* ret);
WP_API void builtin_asin_float16(float16 x, float16* ret);
WP_API void builtin_asin_float32(float32 x, float32* ret);
WP_API void builtin_asin_float64(float64 x, float64* ret);
WP_API void builtin_sqrt_float16(float16 x, float16* ret);
WP_API void builtin_sqrt_float32(float32 x, float32* ret);
WP_API void builtin_sqrt_float64(float64 x, float64* ret);
WP_API void builtin_cbrt_float16(float16 x, float16* ret);
WP_API void builtin_cbrt_float32(float32 x, float32* ret);
WP_API void builtin_cbrt_float64(float64 x, float64* ret);
WP_API void builtin_tan_float16(float16 x, float16* ret);
WP_API void builtin_tan_float32(float32 x, float32* ret);
WP_API void builtin_tan_float64(float64 x, float64* ret);
WP_API void builtin_atan_float16(float16 x, float16* ret);
WP_API void builtin_atan_float32(float32 x, float32* ret);
WP_API void builtin_atan_float64(float64 x, float64* ret);
WP_API void builtin_atan2_float16_float16(float16 y, float16 x, float16* ret);
WP_API void builtin_atan2_float32_float32(float32 y, float32 x, float32* ret);
WP_API void builtin_atan2_float64_float64(float64 y, float64 x, float64* ret);
WP_API void builtin_sinh_float16(float16 x, float16* ret);
WP_API void builtin_sinh_float32(float32 x, float32* ret);
WP_API void builtin_sinh_float64(float64 x, float64* ret);
WP_API void builtin_cosh_float16(float16 x, float16* ret);
WP_API void builtin_cosh_float32(float32 x, float32* ret);
WP_API void builtin_cosh_float64(float64 x, float64* ret);
WP_API void builtin_tanh_float16(float16 x, float16* ret);
WP_API void builtin_tanh_float32(float32 x, float32* ret);
WP_API void builtin_tanh_float64(float64 x, float64* ret);
WP_API void builtin_degrees_float16(float16 x, float16* ret);
WP_API void builtin_degrees_float32(float32 x, float32* ret);
WP_API void builtin_degrees_float64(float64 x, float64* ret);
WP_API void builtin_radians_float16(float16 x, float16* ret);
WP_API void builtin_radians_float32(float32 x, float32* ret);
WP_API void builtin_radians_float64(float64 x, float64* ret);
WP_API void builtin_log_float16(float16 x, float16* ret);
WP_API void builtin_log_float32(float32 x, float32* ret);
WP_API void builtin_log_float64(float64 x, float64* ret);
WP_API void builtin_log2_float16(float16 x, float16* ret);
WP_API void builtin_log2_float32(float32 x, float32* ret);
WP_API void builtin_log2_float64(float64 x, float64* ret);
WP_API void builtin_log10_float16(float16 x, float16* ret);
WP_API void builtin_log10_float32(float32 x, float32* ret);
WP_API void builtin_log10_float64(float64 x, float64* ret);
WP_API void builtin_exp_float16(float16 x, float16* ret);
WP_API void builtin_exp_float32(float32 x, float32* ret);
WP_API void builtin_exp_float64(float64 x, float64* ret);
WP_API void builtin_pow_float16_float16(float16 x, float16 y, float16* ret);
WP_API void builtin_pow_float32_float32(float32 x, float32 y, float32* ret);
WP_API void builtin_pow_float64_float64(float64 x, float64 y, float64* ret);
WP_API void builtin_round_float16(float16 x, float16* ret);
WP_API void builtin_round_float32(float32 x, float32* ret);
WP_API void builtin_round_float64(float64 x, float64* ret);
WP_API void builtin_rint_float16(float16 x, float16* ret);
WP_API void builtin_rint_float32(float32 x, float32* ret);
WP_API void builtin_rint_float64(float64 x, float64* ret);
WP_API void builtin_trunc_float16(float16 x, float16* ret);
WP_API void builtin_trunc_float32(float32 x, float32* ret);
WP_API void builtin_trunc_float64(float64 x, float64* ret);
WP_API void builtin_floor_float16(float16 x, float16* ret);
WP_API void builtin_floor_float32(float32 x, float32* ret);
WP_API void builtin_floor_float64(float64 x, float64* ret);
WP_API void builtin_ceil_float16(float16 x, float16* ret);
WP_API void builtin_ceil_float32(float32 x, float32* ret);
WP_API void builtin_ceil_float64(float64 x, float64* ret);
WP_API void builtin_frac_float16(float16 x, float16* ret);
WP_API void builtin_frac_float32(float32 x, float32* ret);
WP_API void builtin_frac_float64(float64 x, float64* ret);
WP_API void builtin_isfinite_float16(float16 a, bool* ret);
WP_API void builtin_isfinite_float32(float32 a, bool* ret);
WP_API void builtin_isfinite_float64(float64 a, bool* ret);
WP_API void builtin_isfinite_int16(int16 a, bool* ret);
WP_API void builtin_isfinite_int32(int32 a, bool* ret);
WP_API void builtin_isfinite_int64(int64 a, bool* ret);
WP_API void builtin_isfinite_int8(int8 a, bool* ret);
WP_API void builtin_isfinite_uint16(uint16 a, bool* ret);
WP_API void builtin_isfinite_uint32(uint32 a, bool* ret);
WP_API void builtin_isfinite_uint64(uint64 a, bool* ret);
WP_API void builtin_isfinite_uint8(uint8 a, bool* ret);
WP_API void builtin_isfinite_vec2h(vec2h& a, bool* ret);
WP_API void builtin_isfinite_vec3h(vec3h& a, bool* ret);
WP_API void builtin_isfinite_vec4h(vec4h& a, bool* ret);
WP_API void builtin_isfinite_spatial_vectorh(spatial_vectorh& a, bool* ret);
WP_API void builtin_isfinite_vec2f(vec2f& a, bool* ret);
WP_API void builtin_isfinite_vec3f(vec3f& a, bool* ret);
WP_API void builtin_isfinite_vec4f(vec4f& a, bool* ret);
WP_API void builtin_isfinite_spatial_vectorf(spatial_vectorf& a, bool* ret);
WP_API void builtin_isfinite_vec2d(vec2d& a, bool* ret);
WP_API void builtin_isfinite_vec3d(vec3d& a, bool* ret);
WP_API void builtin_isfinite_vec4d(vec4d& a, bool* ret);
WP_API void builtin_isfinite_spatial_vectord(spatial_vectord& a, bool* ret);
WP_API void builtin_isfinite_vec2s(vec2s& a, bool* ret);
WP_API void builtin_isfinite_vec3s(vec3s& a, bool* ret);
WP_API void builtin_isfinite_vec4s(vec4s& a, bool* ret);
WP_API void builtin_isfinite_vec2i(vec2i& a, bool* ret);
WP_API void builtin_isfinite_vec3i(vec3i& a, bool* ret);
WP_API void builtin_isfinite_vec4i(vec4i& a, bool* ret);
WP_API void builtin_isfinite_vec2l(vec2l& a, bool* ret);
WP_API void builtin_isfinite_vec3l(vec3l& a, bool* ret);
WP_API void builtin_isfinite_vec4l(vec4l& a, bool* ret);
WP_API void builtin_isfinite_vec2b(vec2b& a, bool* ret);
WP_API void builtin_isfinite_vec3b(vec3b& a, bool* ret);
WP_API void builtin_isfinite_vec4b(vec4b& a, bool* ret);
WP_API void builtin_isfinite_vec2us(vec2us& a, bool* ret);
WP_API void builtin_isfinite_vec3us(vec3us& a, bool* ret);
WP_API void builtin_isfinite_vec4us(vec4us& a, bool* ret);
WP_API void builtin_isfinite_vec2ui(vec2ui& a, bool* ret);
WP_API void builtin_isfinite_vec3ui(vec3ui& a, bool* ret);
WP_API void builtin_isfinite_vec4ui(vec4ui& a, bool* ret);
WP_API void builtin_isfinite_vec2ul(vec2ul& a, bool* ret);
WP_API void builtin_isfinite_vec3ul(vec3ul& a, bool* ret);
WP_API void builtin_isfinite_vec4ul(vec4ul& a, bool* ret);
WP_API void builtin_isfinite_vec2ub(vec2ub& a, bool* ret);
WP_API void builtin_isfinite_vec3ub(vec3ub& a, bool* ret);
WP_API void builtin_isfinite_vec4ub(vec4ub& a, bool* ret);
WP_API void builtin_isfinite_quath(quath& a, bool* ret);
WP_API void builtin_isfinite_quatf(quatf& a, bool* ret);
WP_API void builtin_isfinite_quatd(quatd& a, bool* ret);
WP_API void builtin_isfinite_mat22h(mat22h& a, bool* ret);
WP_API void builtin_isfinite_mat33h(mat33h& a, bool* ret);
WP_API void builtin_isfinite_mat44h(mat44h& a, bool* ret);
WP_API void builtin_isfinite_spatial_matrixh(spatial_matrixh& a, bool* ret);
WP_API void builtin_isfinite_mat22f(mat22f& a, bool* ret);
WP_API void builtin_isfinite_mat33f(mat33f& a, bool* ret);
WP_API void builtin_isfinite_mat44f(mat44f& a, bool* ret);
WP_API void builtin_isfinite_spatial_matrixf(spatial_matrixf& a, bool* ret);
WP_API void builtin_isfinite_mat22d(mat22d& a, bool* ret);
WP_API void builtin_isfinite_mat33d(mat33d& a, bool* ret);
WP_API void builtin_isfinite_mat44d(mat44d& a, bool* ret);
WP_API void builtin_isfinite_spatial_matrixd(spatial_matrixd& a, bool* ret);
WP_API void builtin_isnan_float16(float16 a, bool* ret);
WP_API void builtin_isnan_float32(float32 a, bool* ret);
WP_API void builtin_isnan_float64(float64 a, bool* ret);
WP_API void builtin_isnan_int16(int16 a, bool* ret);
WP_API void builtin_isnan_int32(int32 a, bool* ret);
WP_API void builtin_isnan_int64(int64 a, bool* ret);
WP_API void builtin_isnan_int8(int8 a, bool* ret);
WP_API void builtin_isnan_uint16(uint16 a, bool* ret);
WP_API void builtin_isnan_uint32(uint32 a, bool* ret);
WP_API void builtin_isnan_uint64(uint64 a, bool* ret);
WP_API void builtin_isnan_uint8(uint8 a, bool* ret);
WP_API void builtin_isnan_vec2h(vec2h& a, bool* ret);
WP_API void builtin_isnan_vec3h(vec3h& a, bool* ret);
WP_API void builtin_isnan_vec4h(vec4h& a, bool* ret);
WP_API void builtin_isnan_spatial_vectorh(spatial_vectorh& a, bool* ret);
WP_API void builtin_isnan_vec2f(vec2f& a, bool* ret);
WP_API void builtin_isnan_vec3f(vec3f& a, bool* ret);
WP_API void builtin_isnan_vec4f(vec4f& a, bool* ret);
WP_API void builtin_isnan_spatial_vectorf(spatial_vectorf& a, bool* ret);
WP_API void builtin_isnan_vec2d(vec2d& a, bool* ret);
WP_API void builtin_isnan_vec3d(vec3d& a, bool* ret);
WP_API void builtin_isnan_vec4d(vec4d& a, bool* ret);
WP_API void builtin_isnan_spatial_vectord(spatial_vectord& a, bool* ret);
WP_API void builtin_isnan_vec2s(vec2s& a, bool* ret);
WP_API void builtin_isnan_vec3s(vec3s& a, bool* ret);
WP_API void builtin_isnan_vec4s(vec4s& a, bool* ret);
WP_API void builtin_isnan_vec2i(vec2i& a, bool* ret);
WP_API void builtin_isnan_vec3i(vec3i& a, bool* ret);
WP_API void builtin_isnan_vec4i(vec4i& a, bool* ret);
WP_API void builtin_isnan_vec2l(vec2l& a, bool* ret);
WP_API void builtin_isnan_vec3l(vec3l& a, bool* ret);
WP_API void builtin_isnan_vec4l(vec4l& a, bool* ret);
WP_API void builtin_isnan_vec2b(vec2b& a, bool* ret);
WP_API void builtin_isnan_vec3b(vec3b& a, bool* ret);
WP_API void builtin_isnan_vec4b(vec4b& a, bool* ret);
WP_API void builtin_isnan_vec2us(vec2us& a, bool* ret);
WP_API void builtin_isnan_vec3us(vec3us& a, bool* ret);
WP_API void builtin_isnan_vec4us(vec4us& a, bool* ret);
WP_API void builtin_isnan_vec2ui(vec2ui& a, bool* ret);
WP_API void builtin_isnan_vec3ui(vec3ui& a, bool* ret);
WP_API void builtin_isnan_vec4ui(vec4ui& a, bool* ret);
WP_API void builtin_isnan_vec2ul(vec2ul& a, bool* ret);
WP_API void builtin_isnan_vec3ul(vec3ul& a, bool* ret);
WP_API void builtin_isnan_vec4ul(vec4ul& a, bool* ret);
WP_API void builtin_isnan_vec2ub(vec2ub& a, bool* ret);
WP_API void builtin_isnan_vec3ub(vec3ub& a, bool* ret);
WP_API void builtin_isnan_vec4ub(vec4ub& a, bool* ret);
WP_API void builtin_isnan_quath(quath& a, bool* ret);
WP_API void builtin_isnan_quatf(quatf& a, bool* ret);
WP_API void builtin_isnan_quatd(quatd& a, bool* ret);
WP_API void builtin_isnan_mat22h(mat22h& a, bool* ret);
WP_API void builtin_isnan_mat33h(mat33h& a, bool* ret);
WP_API void builtin_isnan_mat44h(mat44h& a, bool* ret);
WP_API void builtin_isnan_spatial_matrixh(spatial_matrixh& a, bool* ret);
WP_API void builtin_isnan_mat22f(mat22f& a, bool* ret);
WP_API void builtin_isnan_mat33f(mat33f& a, bool* ret);
WP_API void builtin_isnan_mat44f(mat44f& a, bool* ret);
WP_API void builtin_isnan_spatial_matrixf(spatial_matrixf& a, bool* ret);
WP_API void builtin_isnan_mat22d(mat22d& a, bool* ret);
WP_API void builtin_isnan_mat33d(mat33d& a, bool* ret);
WP_API void builtin_isnan_mat44d(mat44d& a, bool* ret);
WP_API void builtin_isnan_spatial_matrixd(spatial_matrixd& a, bool* ret);
WP_API void builtin_isinf_float16(float16 a, bool* ret);
WP_API void builtin_isinf_float32(float32 a, bool* ret);
WP_API void builtin_isinf_float64(float64 a, bool* ret);
WP_API void builtin_isinf_int16(int16 a, bool* ret);
WP_API void builtin_isinf_int32(int32 a, bool* ret);
WP_API void builtin_isinf_int64(int64 a, bool* ret);
WP_API void builtin_isinf_int8(int8 a, bool* ret);
WP_API void builtin_isinf_uint16(uint16 a, bool* ret);
WP_API void builtin_isinf_uint32(uint32 a, bool* ret);
WP_API void builtin_isinf_uint64(uint64 a, bool* ret);
WP_API void builtin_isinf_uint8(uint8 a, bool* ret);
WP_API void builtin_isinf_vec2h(vec2h& a, bool* ret);
WP_API void builtin_isinf_vec3h(vec3h& a, bool* ret);
WP_API void builtin_isinf_vec4h(vec4h& a, bool* ret);
WP_API void builtin_isinf_spatial_vectorh(spatial_vectorh& a, bool* ret);
WP_API void builtin_isinf_vec2f(vec2f& a, bool* ret);
WP_API void builtin_isinf_vec3f(vec3f& a, bool* ret);
WP_API void builtin_isinf_vec4f(vec4f& a, bool* ret);
WP_API void builtin_isinf_spatial_vectorf(spatial_vectorf& a, bool* ret);
WP_API void builtin_isinf_vec2d(vec2d& a, bool* ret);
WP_API void builtin_isinf_vec3d(vec3d& a, bool* ret);
WP_API void builtin_isinf_vec4d(vec4d& a, bool* ret);
WP_API void builtin_isinf_spatial_vectord(spatial_vectord& a, bool* ret);
WP_API void builtin_isinf_vec2s(vec2s& a, bool* ret);
WP_API void builtin_isinf_vec3s(vec3s& a, bool* ret);
WP_API void builtin_isinf_vec4s(vec4s& a, bool* ret);
WP_API void builtin_isinf_vec2i(vec2i& a, bool* ret);
WP_API void builtin_isinf_vec3i(vec3i& a, bool* ret);
WP_API void builtin_isinf_vec4i(vec4i& a, bool* ret);
WP_API void builtin_isinf_vec2l(vec2l& a, bool* ret);
WP_API void builtin_isinf_vec3l(vec3l& a, bool* ret);
WP_API void builtin_isinf_vec4l(vec4l& a, bool* ret);
WP_API void builtin_isinf_vec2b(vec2b& a, bool* ret);
WP_API void builtin_isinf_vec3b(vec3b& a, bool* ret);
WP_API void builtin_isinf_vec4b(vec4b& a, bool* ret);
WP_API void builtin_isinf_vec2us(vec2us& a, bool* ret);
WP_API void builtin_isinf_vec3us(vec3us& a, bool* ret);
WP_API void builtin_isinf_vec4us(vec4us& a, bool* ret);
WP_API void builtin_isinf_vec2ui(vec2ui& a, bool* ret);
WP_API void builtin_isinf_vec3ui(vec3ui& a, bool* ret);
WP_API void builtin_isinf_vec4ui(vec4ui& a, bool* ret);
WP_API void builtin_isinf_vec2ul(vec2ul& a, bool* ret);
WP_API void builtin_isinf_vec3ul(vec3ul& a, bool* ret);
WP_API void builtin_isinf_vec4ul(vec4ul& a, bool* ret);
WP_API void builtin_isinf_vec2ub(vec2ub& a, bool* ret);
WP_API void builtin_isinf_vec3ub(vec3ub& a, bool* ret);
WP_API void builtin_isinf_vec4ub(vec4ub& a, bool* ret);
WP_API void builtin_isinf_quath(quath& a, bool* ret);
WP_API void builtin_isinf_quatf(quatf& a, bool* ret);
WP_API void builtin_isinf_quatd(quatd& a, bool* ret);
WP_API void builtin_isinf_mat22h(mat22h& a, bool* ret);
WP_API void builtin_isinf_mat33h(mat33h& a, bool* ret);
WP_API void builtin_isinf_mat44h(mat44h& a, bool* ret);
WP_API void builtin_isinf_spatial_matrixh(spatial_matrixh& a, bool* ret);
WP_API void builtin_isinf_mat22f(mat22f& a, bool* ret);
WP_API void builtin_isinf_mat33f(mat33f& a, bool* ret);
WP_API void builtin_isinf_mat44f(mat44f& a, bool* ret);
WP_API void builtin_isinf_spatial_matrixf(spatial_matrixf& a, bool* ret);
WP_API void builtin_isinf_mat22d(mat22d& a, bool* ret);
WP_API void builtin_isinf_mat33d(mat33d& a, bool* ret);
WP_API void builtin_isinf_mat44d(mat44d& a, bool* ret);
WP_API void builtin_isinf_spatial_matrixd(spatial_matrixd& a, bool* ret);
WP_API void builtin_dot_vec2h_vec2h(vec2h& a, vec2h& b, float16* ret);
WP_API void builtin_dot_vec3h_vec3h(vec3h& a, vec3h& b, float16* ret);
WP_API void builtin_dot_vec4h_vec4h(vec4h& a, vec4h& b, float16* ret);
WP_API void builtin_dot_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, float16* ret);
WP_API void builtin_dot_vec2f_vec2f(vec2f& a, vec2f& b, float32* ret);
WP_API void builtin_dot_vec3f_vec3f(vec3f& a, vec3f& b, float32* ret);
WP_API void builtin_dot_vec4f_vec4f(vec4f& a, vec4f& b, float32* ret);
WP_API void builtin_dot_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, float32* ret);
WP_API void builtin_dot_vec2d_vec2d(vec2d& a, vec2d& b, float64* ret);
WP_API void builtin_dot_vec3d_vec3d(vec3d& a, vec3d& b, float64* ret);
WP_API void builtin_dot_vec4d_vec4d(vec4d& a, vec4d& b, float64* ret);
WP_API void builtin_dot_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, float64* ret);
WP_API void builtin_dot_vec2s_vec2s(vec2s& a, vec2s& b, int16* ret);
WP_API void builtin_dot_vec3s_vec3s(vec3s& a, vec3s& b, int16* ret);
WP_API void builtin_dot_vec4s_vec4s(vec4s& a, vec4s& b, int16* ret);
WP_API void builtin_dot_vec2i_vec2i(vec2i& a, vec2i& b, int32* ret);
WP_API void builtin_dot_vec3i_vec3i(vec3i& a, vec3i& b, int32* ret);
WP_API void builtin_dot_vec4i_vec4i(vec4i& a, vec4i& b, int32* ret);
WP_API void builtin_dot_vec2l_vec2l(vec2l& a, vec2l& b, int64* ret);
WP_API void builtin_dot_vec3l_vec3l(vec3l& a, vec3l& b, int64* ret);
WP_API void builtin_dot_vec4l_vec4l(vec4l& a, vec4l& b, int64* ret);
WP_API void builtin_dot_vec2b_vec2b(vec2b& a, vec2b& b, int8* ret);
WP_API void builtin_dot_vec3b_vec3b(vec3b& a, vec3b& b, int8* ret);
WP_API void builtin_dot_vec4b_vec4b(vec4b& a, vec4b& b, int8* ret);
WP_API void builtin_dot_vec2us_vec2us(vec2us& a, vec2us& b, uint16* ret);
WP_API void builtin_dot_vec3us_vec3us(vec3us& a, vec3us& b, uint16* ret);
WP_API void builtin_dot_vec4us_vec4us(vec4us& a, vec4us& b, uint16* ret);
WP_API void builtin_dot_vec2ui_vec2ui(vec2ui& a, vec2ui& b, uint32* ret);
WP_API void builtin_dot_vec3ui_vec3ui(vec3ui& a, vec3ui& b, uint32* ret);
WP_API void builtin_dot_vec4ui_vec4ui(vec4ui& a, vec4ui& b, uint32* ret);
WP_API void builtin_dot_vec2ul_vec2ul(vec2ul& a, vec2ul& b, uint64* ret);
WP_API void builtin_dot_vec3ul_vec3ul(vec3ul& a, vec3ul& b, uint64* ret);
WP_API void builtin_dot_vec4ul_vec4ul(vec4ul& a, vec4ul& b, uint64* ret);
WP_API void builtin_dot_vec2ub_vec2ub(vec2ub& a, vec2ub& b, uint8* ret);
WP_API void builtin_dot_vec3ub_vec3ub(vec3ub& a, vec3ub& b, uint8* ret);
WP_API void builtin_dot_vec4ub_vec4ub(vec4ub& a, vec4ub& b, uint8* ret);
WP_API void builtin_dot_quath_quath(quath& a, quath& b, float16* ret);
WP_API void builtin_dot_quatf_quatf(quatf& a, quatf& b, float32* ret);
WP_API void builtin_dot_quatd_quatd(quatd& a, quatd& b, float64* ret);
WP_API void builtin_ddot_mat22h_mat22h(mat22h& a, mat22h& b, float16* ret);
WP_API void builtin_ddot_mat33h_mat33h(mat33h& a, mat33h& b, float16* ret);
WP_API void builtin_ddot_mat44h_mat44h(mat44h& a, mat44h& b, float16* ret);
WP_API void builtin_ddot_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, float16* ret);
WP_API void builtin_ddot_mat22f_mat22f(mat22f& a, mat22f& b, float32* ret);
WP_API void builtin_ddot_mat33f_mat33f(mat33f& a, mat33f& b, float32* ret);
WP_API void builtin_ddot_mat44f_mat44f(mat44f& a, mat44f& b, float32* ret);
WP_API void builtin_ddot_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, float32* ret);
WP_API void builtin_ddot_mat22d_mat22d(mat22d& a, mat22d& b, float64* ret);
WP_API void builtin_ddot_mat33d_mat33d(mat33d& a, mat33d& b, float64* ret);
WP_API void builtin_ddot_mat44d_mat44d(mat44d& a, mat44d& b, float64* ret);
WP_API void builtin_ddot_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, float64* ret);
WP_API void builtin_argmin_vec2h(vec2h& a, uint32* ret);
WP_API void builtin_argmin_vec3h(vec3h& a, uint32* ret);
WP_API void builtin_argmin_vec4h(vec4h& a, uint32* ret);
WP_API void builtin_argmin_spatial_vectorh(spatial_vectorh& a, uint32* ret);
WP_API void builtin_argmin_vec2f(vec2f& a, uint32* ret);
WP_API void builtin_argmin_vec3f(vec3f& a, uint32* ret);
WP_API void builtin_argmin_vec4f(vec4f& a, uint32* ret);
WP_API void builtin_argmin_spatial_vectorf(spatial_vectorf& a, uint32* ret);
WP_API void builtin_argmin_vec2d(vec2d& a, uint32* ret);
WP_API void builtin_argmin_vec3d(vec3d& a, uint32* ret);
WP_API void builtin_argmin_vec4d(vec4d& a, uint32* ret);
WP_API void builtin_argmin_spatial_vectord(spatial_vectord& a, uint32* ret);
WP_API void builtin_argmin_vec2s(vec2s& a, uint32* ret);
WP_API void builtin_argmin_vec3s(vec3s& a, uint32* ret);
WP_API void builtin_argmin_vec4s(vec4s& a, uint32* ret);
WP_API void builtin_argmin_vec2i(vec2i& a, uint32* ret);
WP_API void builtin_argmin_vec3i(vec3i& a, uint32* ret);
WP_API void builtin_argmin_vec4i(vec4i& a, uint32* ret);
WP_API void builtin_argmin_vec2l(vec2l& a, uint32* ret);
WP_API void builtin_argmin_vec3l(vec3l& a, uint32* ret);
WP_API void builtin_argmin_vec4l(vec4l& a, uint32* ret);
WP_API void builtin_argmin_vec2b(vec2b& a, uint32* ret);
WP_API void builtin_argmin_vec3b(vec3b& a, uint32* ret);
WP_API void builtin_argmin_vec4b(vec4b& a, uint32* ret);
WP_API void builtin_argmin_vec2us(vec2us& a, uint32* ret);
WP_API void builtin_argmin_vec3us(vec3us& a, uint32* ret);
WP_API void builtin_argmin_vec4us(vec4us& a, uint32* ret);
WP_API void builtin_argmin_vec2ui(vec2ui& a, uint32* ret);
WP_API void builtin_argmin_vec3ui(vec3ui& a, uint32* ret);
WP_API void builtin_argmin_vec4ui(vec4ui& a, uint32* ret);
WP_API void builtin_argmin_vec2ul(vec2ul& a, uint32* ret);
WP_API void builtin_argmin_vec3ul(vec3ul& a, uint32* ret);
WP_API void builtin_argmin_vec4ul(vec4ul& a, uint32* ret);
WP_API void builtin_argmin_vec2ub(vec2ub& a, uint32* ret);
WP_API void builtin_argmin_vec3ub(vec3ub& a, uint32* ret);
WP_API void builtin_argmin_vec4ub(vec4ub& a, uint32* ret);
WP_API void builtin_argmax_vec2h(vec2h& a, uint32* ret);
WP_API void builtin_argmax_vec3h(vec3h& a, uint32* ret);
WP_API void builtin_argmax_vec4h(vec4h& a, uint32* ret);
WP_API void builtin_argmax_spatial_vectorh(spatial_vectorh& a, uint32* ret);
WP_API void builtin_argmax_vec2f(vec2f& a, uint32* ret);
WP_API void builtin_argmax_vec3f(vec3f& a, uint32* ret);
WP_API void builtin_argmax_vec4f(vec4f& a, uint32* ret);
WP_API void builtin_argmax_spatial_vectorf(spatial_vectorf& a, uint32* ret);
WP_API void builtin_argmax_vec2d(vec2d& a, uint32* ret);
WP_API void builtin_argmax_vec3d(vec3d& a, uint32* ret);
WP_API void builtin_argmax_vec4d(vec4d& a, uint32* ret);
WP_API void builtin_argmax_spatial_vectord(spatial_vectord& a, uint32* ret);
WP_API void builtin_argmax_vec2s(vec2s& a, uint32* ret);
WP_API void builtin_argmax_vec3s(vec3s& a, uint32* ret);
WP_API void builtin_argmax_vec4s(vec4s& a, uint32* ret);
WP_API void builtin_argmax_vec2i(vec2i& a, uint32* ret);
WP_API void builtin_argmax_vec3i(vec3i& a, uint32* ret);
WP_API void builtin_argmax_vec4i(vec4i& a, uint32* ret);
WP_API void builtin_argmax_vec2l(vec2l& a, uint32* ret);
WP_API void builtin_argmax_vec3l(vec3l& a, uint32* ret);
WP_API void builtin_argmax_vec4l(vec4l& a, uint32* ret);
WP_API void builtin_argmax_vec2b(vec2b& a, uint32* ret);
WP_API void builtin_argmax_vec3b(vec3b& a, uint32* ret);
WP_API void builtin_argmax_vec4b(vec4b& a, uint32* ret);
WP_API void builtin_argmax_vec2us(vec2us& a, uint32* ret);
WP_API void builtin_argmax_vec3us(vec3us& a, uint32* ret);
WP_API void builtin_argmax_vec4us(vec4us& a, uint32* ret);
WP_API void builtin_argmax_vec2ui(vec2ui& a, uint32* ret);
WP_API void builtin_argmax_vec3ui(vec3ui& a, uint32* ret);
WP_API void builtin_argmax_vec4ui(vec4ui& a, uint32* ret);
WP_API void builtin_argmax_vec2ul(vec2ul& a, uint32* ret);
WP_API void builtin_argmax_vec3ul(vec3ul& a, uint32* ret);
WP_API void builtin_argmax_vec4ul(vec4ul& a, uint32* ret);
WP_API void builtin_argmax_vec2ub(vec2ub& a, uint32* ret);
WP_API void builtin_argmax_vec3ub(vec3ub& a, uint32* ret);
WP_API void builtin_argmax_vec4ub(vec4ub& a, uint32* ret);
WP_API void builtin_outer_vec2h_vec2h(vec2h& a, vec2h& b, mat22h* ret);
WP_API void builtin_outer_vec3h_vec3h(vec3h& a, vec3h& b, mat33h* ret);
WP_API void builtin_outer_vec4h_vec4h(vec4h& a, vec4h& b, mat44h* ret);
WP_API void builtin_outer_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_matrixh* ret);
WP_API void builtin_outer_vec2f_vec2f(vec2f& a, vec2f& b, mat22f* ret);
WP_API void builtin_outer_vec3f_vec3f(vec3f& a, vec3f& b, mat33f* ret);
WP_API void builtin_outer_vec4f_vec4f(vec4f& a, vec4f& b, mat44f* ret);
WP_API void builtin_outer_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_matrixf* ret);
WP_API void builtin_outer_vec2d_vec2d(vec2d& a, vec2d& b, mat22d* ret);
WP_API void builtin_outer_vec3d_vec3d(vec3d& a, vec3d& b, mat33d* ret);
WP_API void builtin_outer_vec4d_vec4d(vec4d& a, vec4d& b, mat44d* ret);
WP_API void builtin_outer_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_matrixd* ret);
WP_API void builtin_cross_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret);
WP_API void builtin_cross_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret);
WP_API void builtin_cross_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret);
WP_API void builtin_cross_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret);
WP_API void builtin_cross_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret);
WP_API void builtin_cross_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret);
WP_API void builtin_cross_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret);
WP_API void builtin_cross_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret);
WP_API void builtin_cross_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret);
WP_API void builtin_cross_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret);
WP_API void builtin_cross_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret);
WP_API void builtin_skew_vec3h(vec3h& vec, mat33h* ret);
WP_API void builtin_skew_vec3f(vec3f& vec, mat33f* ret);
WP_API void builtin_skew_vec3d(vec3d& vec, mat33d* ret);
WP_API void builtin_length_vec2h(vec2h& a, float16* ret);
WP_API void builtin_length_vec3h(vec3h& a, float16* ret);
WP_API void builtin_length_vec4h(vec4h& a, float16* ret);
WP_API void builtin_length_spatial_vectorh(spatial_vectorh& a, float16* ret);
WP_API void builtin_length_vec2f(vec2f& a, float32* ret);
WP_API void builtin_length_vec3f(vec3f& a, float32* ret);
WP_API void builtin_length_vec4f(vec4f& a, float32* ret);
WP_API void builtin_length_spatial_vectorf(spatial_vectorf& a, float32* ret);
WP_API void builtin_length_vec2d(vec2d& a, float64* ret);
WP_API void builtin_length_vec3d(vec3d& a, float64* ret);
WP_API void builtin_length_vec4d(vec4d& a, float64* ret);
WP_API void builtin_length_spatial_vectord(spatial_vectord& a, float64* ret);
WP_API void builtin_length_quath(quath& a, float16* ret);
WP_API void builtin_length_quatf(quatf& a, float32* ret);
WP_API void builtin_length_quatd(quatd& a, float64* ret);
WP_API void builtin_length_sq_vec2h(vec2h& a, float16* ret);
WP_API void builtin_length_sq_vec3h(vec3h& a, float16* ret);
WP_API void builtin_length_sq_vec4h(vec4h& a, float16* ret);
WP_API void builtin_length_sq_spatial_vectorh(spatial_vectorh& a, float16* ret);
WP_API void builtin_length_sq_vec2f(vec2f& a, float32* ret);
WP_API void builtin_length_sq_vec3f(vec3f& a, float32* ret);
WP_API void builtin_length_sq_vec4f(vec4f& a, float32* ret);
WP_API void builtin_length_sq_spatial_vectorf(spatial_vectorf& a, float32* ret);
WP_API void builtin_length_sq_vec2d(vec2d& a, float64* ret);
WP_API void builtin_length_sq_vec3d(vec3d& a, float64* ret);
WP_API void builtin_length_sq_vec4d(vec4d& a, float64* ret);
WP_API void builtin_length_sq_spatial_vectord(spatial_vectord& a, float64* ret);
WP_API void builtin_length_sq_vec2s(vec2s& a, int16* ret);
WP_API void builtin_length_sq_vec3s(vec3s& a, int16* ret);
WP_API void builtin_length_sq_vec4s(vec4s& a, int16* ret);
WP_API void builtin_length_sq_vec2i(vec2i& a, int32* ret);
WP_API void builtin_length_sq_vec3i(vec3i& a, int32* ret);
WP_API void builtin_length_sq_vec4i(vec4i& a, int32* ret);
WP_API void builtin_length_sq_vec2l(vec2l& a, int64* ret);
WP_API void builtin_length_sq_vec3l(vec3l& a, int64* ret);
WP_API void builtin_length_sq_vec4l(vec4l& a, int64* ret);
WP_API void builtin_length_sq_vec2b(vec2b& a, int8* ret);
WP_API void builtin_length_sq_vec3b(vec3b& a, int8* ret);
WP_API void builtin_length_sq_vec4b(vec4b& a, int8* ret);
WP_API void builtin_length_sq_vec2us(vec2us& a, uint16* ret);
WP_API void builtin_length_sq_vec3us(vec3us& a, uint16* ret);
WP_API void builtin_length_sq_vec4us(vec4us& a, uint16* ret);
WP_API void builtin_length_sq_vec2ui(vec2ui& a, uint32* ret);
WP_API void builtin_length_sq_vec3ui(vec3ui& a, uint32* ret);
WP_API void builtin_length_sq_vec4ui(vec4ui& a, uint32* ret);
WP_API void builtin_length_sq_vec2ul(vec2ul& a, uint64* ret);
WP_API void builtin_length_sq_vec3ul(vec3ul& a, uint64* ret);
WP_API void builtin_length_sq_vec4ul(vec4ul& a, uint64* ret);
WP_API void builtin_length_sq_vec2ub(vec2ub& a, uint8* ret);
WP_API void builtin_length_sq_vec3ub(vec3ub& a, uint8* ret);
WP_API void builtin_length_sq_vec4ub(vec4ub& a, uint8* ret);
WP_API void builtin_length_sq_quath(quath& a, float16* ret);
WP_API void builtin_length_sq_quatf(quatf& a, float32* ret);
WP_API void builtin_length_sq_quatd(quatd& a, float64* ret);
WP_API void builtin_normalize_vec2h(vec2h& a, vec2h* ret);
WP_API void builtin_normalize_vec3h(vec3h& a, vec3h* ret);
WP_API void builtin_normalize_vec4h(vec4h& a, vec4h* ret);
WP_API void builtin_normalize_spatial_vectorh(spatial_vectorh& a, spatial_vectorh* ret);
WP_API void builtin_normalize_vec2f(vec2f& a, vec2f* ret);
WP_API void builtin_normalize_vec3f(vec3f& a, vec3f* ret);
WP_API void builtin_normalize_vec4f(vec4f& a, vec4f* ret);
WP_API void builtin_normalize_spatial_vectorf(spatial_vectorf& a, spatial_vectorf* ret);
WP_API void builtin_normalize_vec2d(vec2d& a, vec2d* ret);
WP_API void builtin_normalize_vec3d(vec3d& a, vec3d* ret);
WP_API void builtin_normalize_vec4d(vec4d& a, vec4d* ret);
WP_API void builtin_normalize_spatial_vectord(spatial_vectord& a, spatial_vectord* ret);
WP_API void builtin_normalize_quath(quath& a, quath* ret);
WP_API void builtin_normalize_quatf(quatf& a, quatf* ret);
WP_API void builtin_normalize_quatd(quatd& a, quatd* ret);
WP_API void builtin_transpose_mat22h(mat22h& a, mat22h* ret);
WP_API void builtin_transpose_mat33h(mat33h& a, mat33h* ret);
WP_API void builtin_transpose_mat44h(mat44h& a, mat44h* ret);
WP_API void builtin_transpose_spatial_matrixh(spatial_matrixh& a, spatial_matrixh* ret);
WP_API void builtin_transpose_mat22f(mat22f& a, mat22f* ret);
WP_API void builtin_transpose_mat33f(mat33f& a, mat33f* ret);
WP_API void builtin_transpose_mat44f(mat44f& a, mat44f* ret);
WP_API void builtin_transpose_spatial_matrixf(spatial_matrixf& a, spatial_matrixf* ret);
WP_API void builtin_transpose_mat22d(mat22d& a, mat22d* ret);
WP_API void builtin_transpose_mat33d(mat33d& a, mat33d* ret);
WP_API void builtin_transpose_mat44d(mat44d& a, mat44d* ret);
WP_API void builtin_transpose_spatial_matrixd(spatial_matrixd& a, spatial_matrixd* ret);
WP_API void builtin_inverse_mat22h(mat22h& a, mat22h* ret);
WP_API void builtin_inverse_mat22f(mat22f& a, mat22f* ret);
WP_API void builtin_inverse_mat22d(mat22d& a, mat22d* ret);
WP_API void builtin_inverse_mat33h(mat33h& a, mat33h* ret);
WP_API void builtin_inverse_mat33f(mat33f& a, mat33f* ret);
WP_API void builtin_inverse_mat33d(mat33d& a, mat33d* ret);
WP_API void builtin_inverse_mat44h(mat44h& a, mat44h* ret);
WP_API void builtin_inverse_mat44f(mat44f& a, mat44f* ret);
WP_API void builtin_inverse_mat44d(mat44d& a, mat44d* ret);
WP_API void builtin_determinant_mat22h(mat22h& a, float16* ret);
WP_API void builtin_determinant_mat22f(mat22f& a, float32* ret);
WP_API void builtin_determinant_mat22d(mat22d& a, float64* ret);
WP_API void builtin_determinant_mat33h(mat33h& a, float16* ret);
WP_API void builtin_determinant_mat33f(mat33f& a, float32* ret);
WP_API void builtin_determinant_mat33d(mat33d& a, float64* ret);
WP_API void builtin_determinant_mat44h(mat44h& a, float16* ret);
WP_API void builtin_determinant_mat44f(mat44f& a, float32* ret);
WP_API void builtin_determinant_mat44d(mat44d& a, float64* ret);
WP_API void builtin_trace_mat22h(mat22h& a, float16* ret);
WP_API void builtin_trace_mat33h(mat33h& a, float16* ret);
WP_API void builtin_trace_mat44h(mat44h& a, float16* ret);
WP_API void builtin_trace_spatial_matrixh(spatial_matrixh& a, float16* ret);
WP_API void builtin_trace_mat22f(mat22f& a, float32* ret);
WP_API void builtin_trace_mat33f(mat33f& a, float32* ret);
WP_API void builtin_trace_mat44f(mat44f& a, float32* ret);
WP_API void builtin_trace_spatial_matrixf(spatial_matrixf& a, float32* ret);
WP_API void builtin_trace_mat22d(mat22d& a, float64* ret);
WP_API void builtin_trace_mat33d(mat33d& a, float64* ret);
WP_API void builtin_trace_mat44d(mat44d& a, float64* ret);
WP_API void builtin_trace_spatial_matrixd(spatial_matrixd& a, float64* ret);
WP_API void builtin_diag_vec2h(vec2h& vec, mat22h* ret);
WP_API void builtin_diag_vec3h(vec3h& vec, mat33h* ret);
WP_API void builtin_diag_vec4h(vec4h& vec, mat44h* ret);
WP_API void builtin_diag_spatial_vectorh(spatial_vectorh& vec, spatial_matrixh* ret);
WP_API void builtin_diag_vec2f(vec2f& vec, mat22f* ret);
WP_API void builtin_diag_vec3f(vec3f& vec, mat33f* ret);
WP_API void builtin_diag_vec4f(vec4f& vec, mat44f* ret);
WP_API void builtin_diag_spatial_vectorf(spatial_vectorf& vec, spatial_matrixf* ret);
WP_API void builtin_diag_vec2d(vec2d& vec, mat22d* ret);
WP_API void builtin_diag_vec3d(vec3d& vec, mat33d* ret);
WP_API void builtin_diag_vec4d(vec4d& vec, mat44d* ret);
WP_API void builtin_diag_spatial_vectord(spatial_vectord& vec, spatial_matrixd* ret);
WP_API void builtin_get_diag_mat22h(mat22h& mat, vec2h* ret);
WP_API void builtin_get_diag_mat33h(mat33h& mat, vec3h* ret);
WP_API void builtin_get_diag_mat44h(mat44h& mat, vec4h* ret);
WP_API void builtin_get_diag_spatial_matrixh(spatial_matrixh& mat, spatial_vectorh* ret);
WP_API void builtin_get_diag_mat22f(mat22f& mat, vec2f* ret);
WP_API void builtin_get_diag_mat33f(mat33f& mat, vec3f* ret);
WP_API void builtin_get_diag_mat44f(mat44f& mat, vec4f* ret);
WP_API void builtin_get_diag_spatial_matrixf(spatial_matrixf& mat, spatial_vectorf* ret);
WP_API void builtin_get_diag_mat22d(mat22d& mat, vec2d* ret);
WP_API void builtin_get_diag_mat33d(mat33d& mat, vec3d* ret);
WP_API void builtin_get_diag_mat44d(mat44d& mat, vec4d* ret);
WP_API void builtin_get_diag_spatial_matrixd(spatial_matrixd& mat, spatial_vectord* ret);
WP_API void builtin_cw_mul_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret);
WP_API void builtin_cw_mul_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret);
WP_API void builtin_cw_mul_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret);
WP_API void builtin_cw_mul_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_cw_mul_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret);
WP_API void builtin_cw_mul_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret);
WP_API void builtin_cw_mul_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret);
WP_API void builtin_cw_mul_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_cw_mul_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret);
WP_API void builtin_cw_mul_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret);
WP_API void builtin_cw_mul_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret);
WP_API void builtin_cw_mul_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_cw_mul_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret);
WP_API void builtin_cw_mul_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret);
WP_API void builtin_cw_mul_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret);
WP_API void builtin_cw_mul_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret);
WP_API void builtin_cw_mul_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret);
WP_API void builtin_cw_mul_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret);
WP_API void builtin_cw_mul_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret);
WP_API void builtin_cw_mul_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret);
WP_API void builtin_cw_mul_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret);
WP_API void builtin_cw_mul_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret);
WP_API void builtin_cw_mul_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret);
WP_API void builtin_cw_mul_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret);
WP_API void builtin_cw_mul_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret);
WP_API void builtin_cw_mul_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret);
WP_API void builtin_cw_mul_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret);
WP_API void builtin_cw_mul_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret);
WP_API void builtin_cw_mul_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret);
WP_API void builtin_cw_mul_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret);
WP_API void builtin_cw_mul_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret);
WP_API void builtin_cw_mul_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret);
WP_API void builtin_cw_mul_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret);
WP_API void builtin_cw_mul_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret);
WP_API void builtin_cw_mul_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret);
WP_API void builtin_cw_mul_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret);
WP_API void builtin_cw_mul_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret);
WP_API void builtin_cw_mul_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret);
WP_API void builtin_cw_mul_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret);
WP_API void builtin_cw_mul_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret);
WP_API void builtin_cw_mul_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret);
WP_API void builtin_cw_mul_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret);
WP_API void builtin_cw_mul_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret);
WP_API void builtin_cw_mul_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret);
WP_API void builtin_cw_mul_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret);
WP_API void builtin_cw_mul_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret);
WP_API void builtin_cw_mul_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret);
WP_API void builtin_cw_mul_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret);
WP_API void builtin_cw_div_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret);
WP_API void builtin_cw_div_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret);
WP_API void builtin_cw_div_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret);
WP_API void builtin_cw_div_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_cw_div_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret);
WP_API void builtin_cw_div_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret);
WP_API void builtin_cw_div_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret);
WP_API void builtin_cw_div_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_cw_div_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret);
WP_API void builtin_cw_div_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret);
WP_API void builtin_cw_div_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret);
WP_API void builtin_cw_div_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_cw_div_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret);
WP_API void builtin_cw_div_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret);
WP_API void builtin_cw_div_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret);
WP_API void builtin_cw_div_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret);
WP_API void builtin_cw_div_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret);
WP_API void builtin_cw_div_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret);
WP_API void builtin_cw_div_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret);
WP_API void builtin_cw_div_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret);
WP_API void builtin_cw_div_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret);
WP_API void builtin_cw_div_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret);
WP_API void builtin_cw_div_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret);
WP_API void builtin_cw_div_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret);
WP_API void builtin_cw_div_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret);
WP_API void builtin_cw_div_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret);
WP_API void builtin_cw_div_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret);
WP_API void builtin_cw_div_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret);
WP_API void builtin_cw_div_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret);
WP_API void builtin_cw_div_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret);
WP_API void builtin_cw_div_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret);
WP_API void builtin_cw_div_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret);
WP_API void builtin_cw_div_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret);
WP_API void builtin_cw_div_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret);
WP_API void builtin_cw_div_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret);
WP_API void builtin_cw_div_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret);
WP_API void builtin_cw_div_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret);
WP_API void builtin_cw_div_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret);
WP_API void builtin_cw_div_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret);
WP_API void builtin_cw_div_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret);
WP_API void builtin_cw_div_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret);
WP_API void builtin_cw_div_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret);
WP_API void builtin_cw_div_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret);
WP_API void builtin_cw_div_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret);
WP_API void builtin_cw_div_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret);
WP_API void builtin_cw_div_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret);
WP_API void builtin_cw_div_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret);
WP_API void builtin_cw_div_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret);
WP_API void builtin_quat_identity_float16(quath* ret);
WP_API void builtin_quat_identity_float32(quatf* ret);
WP_API void builtin_quat_identity_float64(quatd* ret);
WP_API void builtin_quat_from_axis_angle_vec3h_float16(vec3h& axis, float16 angle, quath* ret);
WP_API void builtin_quat_from_axis_angle_vec3f_float32(vec3f& axis, float32 angle, quatf* ret);
WP_API void builtin_quat_from_axis_angle_vec3d_float64(vec3d& axis, float64 angle, quatd* ret);
WP_API void builtin_quat_from_matrix_mat33h(mat33h& mat, quath* ret);
WP_API void builtin_quat_from_matrix_mat33f(mat33f& mat, quatf* ret);
WP_API void builtin_quat_from_matrix_mat33d(mat33d& mat, quatd* ret);
WP_API void builtin_quat_rpy_float16_float16_float16(float16 roll, float16 pitch, float16 yaw, quath* ret);
WP_API void builtin_quat_rpy_float32_float32_float32(float32 roll, float32 pitch, float32 yaw, quatf* ret);
WP_API void builtin_quat_rpy_float64_float64_float64(float64 roll, float64 pitch, float64 yaw, quatd* ret);
WP_API void builtin_quat_inverse_quath(quath& quat, quath* ret);
WP_API void builtin_quat_inverse_quatf(quatf& quat, quatf* ret);
WP_API void builtin_quat_inverse_quatd(quatd& quat, quatd* ret);
WP_API void builtin_quat_rotate_quath_vec3h(quath& quat, vec3h& vec, vec3h* ret);
WP_API void builtin_quat_rotate_quatf_vec3f(quatf& quat, vec3f& vec, vec3f* ret);
WP_API void builtin_quat_rotate_quatd_vec3d(quatd& quat, vec3d& vec, vec3d* ret);
WP_API void builtin_quat_rotate_inv_quath_vec3h(quath& quat, vec3h& vec, vec3h* ret);
WP_API void builtin_quat_rotate_inv_quatf_vec3f(quatf& quat, vec3f& vec, vec3f* ret);
WP_API void builtin_quat_rotate_inv_quatd_vec3d(quatd& quat, vec3d& vec, vec3d* ret);
WP_API void builtin_quat_slerp_quath_quath_float16(quath& a, quath& b, float16 t, quath* ret);
WP_API void builtin_quat_slerp_quatf_quatf_float32(quatf& a, quatf& b, float32 t, quatf* ret);
WP_API void builtin_quat_slerp_quatd_quatd_float64(quatd& a, quatd& b, float64 t, quatd* ret);
WP_API void builtin_quat_to_matrix_quath(quath& quat, mat33h* ret);
WP_API void builtin_quat_to_matrix_quatf(quatf& quat, mat33f* ret);
WP_API void builtin_quat_to_matrix_quatd(quatd& quat, mat33d* ret);
WP_API void builtin_transform_identity_float16(transformh* ret);
WP_API void builtin_transform_identity_float32(transformf* ret);
WP_API void builtin_transform_identity_float64(transformd* ret);
WP_API void builtin_transform_get_translation_transformh(transformh& xform, vec3h* ret);
WP_API void builtin_transform_get_translation_transformf(transformf& xform, vec3f* ret);
WP_API void builtin_transform_get_translation_transformd(transformd& xform, vec3d* ret);
WP_API void builtin_transform_get_rotation_transformh(transformh& xform, quath* ret);
WP_API void builtin_transform_get_rotation_transformf(transformf& xform, quatf* ret);
WP_API void builtin_transform_get_rotation_transformd(transformd& xform, quatd* ret);
WP_API void builtin_transform_multiply_transformh_transformh(transformh& a, transformh& b, transformh* ret);
WP_API void builtin_transform_multiply_transformf_transformf(transformf& a, transformf& b, transformf* ret);
WP_API void builtin_transform_multiply_transformd_transformd(transformd& a, transformd& b, transformd* ret);
WP_API void builtin_transform_point_transformh_vec3h(transformh& xform, vec3h& point, vec3h* ret);
WP_API void builtin_transform_point_transformf_vec3f(transformf& xform, vec3f& point, vec3f* ret);
WP_API void builtin_transform_point_transformd_vec3d(transformd& xform, vec3d& point, vec3d* ret);
WP_API void builtin_transform_point_mat44h_vec3h(mat44h& mat, vec3h& point, vec3h* ret);
WP_API void builtin_transform_point_mat44f_vec3f(mat44f& mat, vec3f& point, vec3f* ret);
WP_API void builtin_transform_point_mat44d_vec3d(mat44d& mat, vec3d& point, vec3d* ret);
WP_API void builtin_transform_vector_transformh_vec3h(transformh& xform, vec3h& vec, vec3h* ret);
WP_API void builtin_transform_vector_transformf_vec3f(transformf& xform, vec3f& vec, vec3f* ret);
WP_API void builtin_transform_vector_transformd_vec3d(transformd& xform, vec3d& vec, vec3d* ret);
WP_API void builtin_transform_vector_mat44h_vec3h(mat44h& mat, vec3h& vec, vec3h* ret);
WP_API void builtin_transform_vector_mat44f_vec3f(mat44f& mat, vec3f& vec, vec3f* ret);
WP_API void builtin_transform_vector_mat44d_vec3d(mat44d& mat, vec3d& vec, vec3d* ret);
WP_API void builtin_transform_inverse_transformh(transformh& xform, transformh* ret);
WP_API void builtin_transform_inverse_transformf(transformf& xform, transformf* ret);
WP_API void builtin_transform_inverse_transformd(transformd& xform, transformd* ret);
WP_API void builtin_spatial_dot_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, float16* ret);
WP_API void builtin_spatial_dot_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, float32* ret);
WP_API void builtin_spatial_dot_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, float64* ret);
WP_API void builtin_spatial_cross_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_spatial_cross_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_spatial_cross_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_spatial_cross_dual_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_spatial_cross_dual_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_spatial_cross_dual_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_spatial_top_spatial_vectorh(spatial_vectorh& svec, vec3h* ret);
WP_API void builtin_spatial_top_spatial_vectorf(spatial_vectorf& svec, vec3f* ret);
WP_API void builtin_spatial_top_spatial_vectord(spatial_vectord& svec, vec3d* ret);
WP_API void builtin_spatial_bottom_spatial_vectorh(spatial_vectorh& svec, vec3h* ret);
WP_API void builtin_spatial_bottom_spatial_vectorf(spatial_vectorf& svec, vec3f* ret);
WP_API void builtin_spatial_bottom_spatial_vectord(spatial_vectord& svec, vec3d* ret);
WP_API void builtin_volume_sample_f_uint64_vec3f_int32(uint64 id, vec3f& uvw, int32 sampling_mode, float* ret);
WP_API void builtin_volume_sample_grad_f_uint64_vec3f_int32_vec3f(uint64 id, vec3f& uvw, int32 sampling_mode, vec3f& grad, float* ret);
WP_API void builtin_volume_lookup_f_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, float* ret);
WP_API void builtin_volume_sample_v_uint64_vec3f_int32(uint64 id, vec3f& uvw, int32 sampling_mode, vec3f* ret);
WP_API void builtin_volume_lookup_v_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, vec3f* ret);
WP_API void builtin_volume_sample_i_uint64_vec3f(uint64 id, vec3f& uvw, int* ret);
WP_API void builtin_volume_lookup_i_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, int* ret);
WP_API void builtin_volume_lookup_index_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, int32* ret);
WP_API void builtin_volume_index_to_world_uint64_vec3f(uint64 id, vec3f& uvw, vec3f* ret);
WP_API void builtin_volume_world_to_index_uint64_vec3f(uint64 id, vec3f& xyz, vec3f* ret);
WP_API void builtin_volume_index_to_world_dir_uint64_vec3f(uint64 id, vec3f& uvw, vec3f* ret);
WP_API void builtin_volume_world_to_index_dir_uint64_vec3f(uint64 id, vec3f& xyz, vec3f* ret);
WP_API void builtin_rand_init_int32(int32 seed, uint32* ret);
WP_API void builtin_rand_init_int32_int32(int32 seed, int32 offset, uint32* ret);
WP_API void builtin_randi_uint32(uint32 state, int* ret);
WP_API void builtin_randi_uint32_int32_int32(uint32 state, int32 low, int32 high, int* ret);
WP_API void builtin_randf_uint32(uint32 state, float* ret);
WP_API void builtin_randf_uint32_float32_float32(uint32 state, float32 low, float32 high, float* ret);
WP_API void builtin_randn_uint32(uint32 state, float* ret);
WP_API void builtin_sample_triangle_uint32(uint32 state, vec2f* ret);
WP_API void builtin_sample_unit_ring_uint32(uint32 state, vec2f* ret);
WP_API void builtin_sample_unit_disk_uint32(uint32 state, vec2f* ret);
WP_API void builtin_sample_unit_sphere_surface_uint32(uint32 state, vec3f* ret);
WP_API void builtin_sample_unit_sphere_uint32(uint32 state, vec3f* ret);
WP_API void builtin_sample_unit_hemisphere_surface_uint32(uint32 state, vec3f* ret);
WP_API void builtin_sample_unit_hemisphere_uint32(uint32 state, vec3f* ret);
WP_API void builtin_sample_unit_square_uint32(uint32 state, vec2f* ret);
WP_API void builtin_sample_unit_cube_uint32(uint32 state, vec3f* ret);
WP_API void builtin_poisson_uint32_float32(uint32 state, float32 lam, uint32* ret);
WP_API void builtin_noise_uint32_float32(uint32 state, float32 x, float* ret);
WP_API void builtin_noise_uint32_vec2f(uint32 state, vec2f& xy, float* ret);
WP_API void builtin_noise_uint32_vec3f(uint32 state, vec3f& xyz, float* ret);
WP_API void builtin_noise_uint32_vec4f(uint32 state, vec4f& xyzt, float* ret);
WP_API void builtin_pnoise_uint32_float32_int32(uint32 state, float32 x, int32 px, float* ret);
WP_API void builtin_pnoise_uint32_vec2f_int32_int32(uint32 state, vec2f& xy, int32 px, int32 py, float* ret);
WP_API void builtin_pnoise_uint32_vec3f_int32_int32_int32(uint32 state, vec3f& xyz, int32 px, int32 py, int32 pz, float* ret);
WP_API void builtin_pnoise_uint32_vec4f_int32_int32_int32_int32(uint32 state, vec4f& xyzt, int32 px, int32 py, int32 pz, int32 pt, float* ret);
WP_API void builtin_curlnoise_uint32_vec2f_uint32_float32_float32(uint32 state, vec2f& xy, uint32 octaves, float32 lacunarity, float32 gain, vec2f* ret);
WP_API void builtin_curlnoise_uint32_vec3f_uint32_float32_float32(uint32 state, vec3f& xyz, uint32 octaves, float32 lacunarity, float32 gain, vec3f* ret);
WP_API void builtin_curlnoise_uint32_vec4f_uint32_float32_float32(uint32 state, vec4f& xyzt, uint32 octaves, float32 lacunarity, float32 gain, vec3f* ret);
WP_API void builtin_assign_vec2h_int32_float16(vec2h& a, int32 i, float16 value, vec2h* ret);
WP_API void builtin_assign_vec3h_int32_float16(vec3h& a, int32 i, float16 value, vec3h* ret);
WP_API void builtin_assign_vec4h_int32_float16(vec4h& a, int32 i, float16 value, vec4h* ret);
WP_API void builtin_assign_spatial_vectorh_int32_float16(spatial_vectorh& a, int32 i, float16 value, spatial_vectorh* ret);
WP_API void builtin_assign_vec2f_int32_float32(vec2f& a, int32 i, float32 value, vec2f* ret);
WP_API void builtin_assign_vec3f_int32_float32(vec3f& a, int32 i, float32 value, vec3f* ret);
WP_API void builtin_assign_vec4f_int32_float32(vec4f& a, int32 i, float32 value, vec4f* ret);
WP_API void builtin_assign_spatial_vectorf_int32_float32(spatial_vectorf& a, int32 i, float32 value, spatial_vectorf* ret);
WP_API void builtin_assign_vec2d_int32_float64(vec2d& a, int32 i, float64 value, vec2d* ret);
WP_API void builtin_assign_vec3d_int32_float64(vec3d& a, int32 i, float64 value, vec3d* ret);
WP_API void builtin_assign_vec4d_int32_float64(vec4d& a, int32 i, float64 value, vec4d* ret);
WP_API void builtin_assign_spatial_vectord_int32_float64(spatial_vectord& a, int32 i, float64 value, spatial_vectord* ret);
WP_API void builtin_assign_vec2s_int32_int16(vec2s& a, int32 i, int16 value, vec2s* ret);
WP_API void builtin_assign_vec3s_int32_int16(vec3s& a, int32 i, int16 value, vec3s* ret);
WP_API void builtin_assign_vec4s_int32_int16(vec4s& a, int32 i, int16 value, vec4s* ret);
WP_API void builtin_assign_vec2i_int32_int32(vec2i& a, int32 i, int32 value, vec2i* ret);
WP_API void builtin_assign_vec3i_int32_int32(vec3i& a, int32 i, int32 value, vec3i* ret);
WP_API void builtin_assign_vec4i_int32_int32(vec4i& a, int32 i, int32 value, vec4i* ret);
WP_API void builtin_assign_vec2l_int32_int64(vec2l& a, int32 i, int64 value, vec2l* ret);
WP_API void builtin_assign_vec3l_int32_int64(vec3l& a, int32 i, int64 value, vec3l* ret);
WP_API void builtin_assign_vec4l_int32_int64(vec4l& a, int32 i, int64 value, vec4l* ret);
WP_API void builtin_assign_vec2b_int32_int8(vec2b& a, int32 i, int8 value, vec2b* ret);
WP_API void builtin_assign_vec3b_int32_int8(vec3b& a, int32 i, int8 value, vec3b* ret);
WP_API void builtin_assign_vec4b_int32_int8(vec4b& a, int32 i, int8 value, vec4b* ret);
WP_API void builtin_assign_vec2us_int32_uint16(vec2us& a, int32 i, uint16 value, vec2us* ret);
WP_API void builtin_assign_vec3us_int32_uint16(vec3us& a, int32 i, uint16 value, vec3us* ret);
WP_API void builtin_assign_vec4us_int32_uint16(vec4us& a, int32 i, uint16 value, vec4us* ret);
WP_API void builtin_assign_vec2ui_int32_uint32(vec2ui& a, int32 i, uint32 value, vec2ui* ret);
WP_API void builtin_assign_vec3ui_int32_uint32(vec3ui& a, int32 i, uint32 value, vec3ui* ret);
WP_API void builtin_assign_vec4ui_int32_uint32(vec4ui& a, int32 i, uint32 value, vec4ui* ret);
WP_API void builtin_assign_vec2ul_int32_uint64(vec2ul& a, int32 i, uint64 value, vec2ul* ret);
WP_API void builtin_assign_vec3ul_int32_uint64(vec3ul& a, int32 i, uint64 value, vec3ul* ret);
WP_API void builtin_assign_vec4ul_int32_uint64(vec4ul& a, int32 i, uint64 value, vec4ul* ret);
WP_API void builtin_assign_vec2ub_int32_uint8(vec2ub& a, int32 i, uint8 value, vec2ub* ret);
WP_API void builtin_assign_vec3ub_int32_uint8(vec3ub& a, int32 i, uint8 value, vec3ub* ret);
WP_API void builtin_assign_vec4ub_int32_uint8(vec4ub& a, int32 i, uint8 value, vec4ub* ret);
WP_API void builtin_assign_quath_int32_float16(quath& a, int32 i, float16 value, quath* ret);
WP_API void builtin_assign_quatf_int32_float32(quatf& a, int32 i, float32 value, quatf* ret);
WP_API void builtin_assign_quatd_int32_float64(quatd& a, int32 i, float64 value, quatd* ret);
WP_API void builtin_assign_mat22h_int32_int32_float16(mat22h& a, int32 i, int32 j, float16 value, mat22h* ret);
WP_API void builtin_assign_mat33h_int32_int32_float16(mat33h& a, int32 i, int32 j, float16 value, mat33h* ret);
WP_API void builtin_assign_mat44h_int32_int32_float16(mat44h& a, int32 i, int32 j, float16 value, mat44h* ret);
WP_API void builtin_assign_spatial_matrixh_int32_int32_float16(spatial_matrixh& a, int32 i, int32 j, float16 value, spatial_matrixh* ret);
WP_API void builtin_assign_mat22f_int32_int32_float32(mat22f& a, int32 i, int32 j, float32 value, mat22f* ret);
WP_API void builtin_assign_mat33f_int32_int32_float32(mat33f& a, int32 i, int32 j, float32 value, mat33f* ret);
WP_API void builtin_assign_mat44f_int32_int32_float32(mat44f& a, int32 i, int32 j, float32 value, mat44f* ret);
WP_API void builtin_assign_spatial_matrixf_int32_int32_float32(spatial_matrixf& a, int32 i, int32 j, float32 value, spatial_matrixf* ret);
WP_API void builtin_assign_mat22d_int32_int32_float64(mat22d& a, int32 i, int32 j, float64 value, mat22d* ret);
WP_API void builtin_assign_mat33d_int32_int32_float64(mat33d& a, int32 i, int32 j, float64 value, mat33d* ret);
WP_API void builtin_assign_mat44d_int32_int32_float64(mat44d& a, int32 i, int32 j, float64 value, mat44d* ret);
WP_API void builtin_assign_spatial_matrixd_int32_int32_float64(spatial_matrixd& a, int32 i, int32 j, float64 value, spatial_matrixd* ret);
WP_API void builtin_assign_mat22h_int32_vec2h(mat22h& a, int32 i, vec2h& value, mat22h* ret);
WP_API void builtin_assign_mat33h_int32_vec3h(mat33h& a, int32 i, vec3h& value, mat33h* ret);
WP_API void builtin_assign_mat44h_int32_vec4h(mat44h& a, int32 i, vec4h& value, mat44h* ret);
WP_API void builtin_assign_spatial_matrixh_int32_spatial_vectorh(spatial_matrixh& a, int32 i, spatial_vectorh& value, spatial_matrixh* ret);
WP_API void builtin_assign_mat22f_int32_vec2f(mat22f& a, int32 i, vec2f& value, mat22f* ret);
WP_API void builtin_assign_mat33f_int32_vec3f(mat33f& a, int32 i, vec3f& value, mat33f* ret);
WP_API void builtin_assign_mat44f_int32_vec4f(mat44f& a, int32 i, vec4f& value, mat44f* ret);
WP_API void builtin_assign_spatial_matrixf_int32_spatial_vectorf(spatial_matrixf& a, int32 i, spatial_vectorf& value, spatial_matrixf* ret);
WP_API void builtin_assign_mat22d_int32_vec2d(mat22d& a, int32 i, vec2d& value, mat22d* ret);
WP_API void builtin_assign_mat33d_int32_vec3d(mat33d& a, int32 i, vec3d& value, mat33d* ret);
WP_API void builtin_assign_mat44d_int32_vec4d(mat44d& a, int32 i, vec4d& value, mat44d* ret);
WP_API void builtin_assign_spatial_matrixd_int32_spatial_vectord(spatial_matrixd& a, int32 i, spatial_vectord& value, spatial_matrixd* ret);
WP_API void builtin_extract_vec2h_int32(vec2h& a, int32 i, float16* ret);
WP_API void builtin_extract_vec3h_int32(vec3h& a, int32 i, float16* ret);
WP_API void builtin_extract_vec4h_int32(vec4h& a, int32 i, float16* ret);
WP_API void builtin_extract_spatial_vectorh_int32(spatial_vectorh& a, int32 i, float16* ret);
WP_API void builtin_extract_vec2f_int32(vec2f& a, int32 i, float32* ret);
WP_API void builtin_extract_vec3f_int32(vec3f& a, int32 i, float32* ret);
WP_API void builtin_extract_vec4f_int32(vec4f& a, int32 i, float32* ret);
WP_API void builtin_extract_spatial_vectorf_int32(spatial_vectorf& a, int32 i, float32* ret);
WP_API void builtin_extract_vec2d_int32(vec2d& a, int32 i, float64* ret);
WP_API void builtin_extract_vec3d_int32(vec3d& a, int32 i, float64* ret);
WP_API void builtin_extract_vec4d_int32(vec4d& a, int32 i, float64* ret);
WP_API void builtin_extract_spatial_vectord_int32(spatial_vectord& a, int32 i, float64* ret);
WP_API void builtin_extract_vec2s_int32(vec2s& a, int32 i, int16* ret);
WP_API void builtin_extract_vec3s_int32(vec3s& a, int32 i, int16* ret);
WP_API void builtin_extract_vec4s_int32(vec4s& a, int32 i, int16* ret);
WP_API void builtin_extract_vec2i_int32(vec2i& a, int32 i, int32* ret);
WP_API void builtin_extract_vec3i_int32(vec3i& a, int32 i, int32* ret);
WP_API void builtin_extract_vec4i_int32(vec4i& a, int32 i, int32* ret);
WP_API void builtin_extract_vec2l_int32(vec2l& a, int32 i, int64* ret);
WP_API void builtin_extract_vec3l_int32(vec3l& a, int32 i, int64* ret);
WP_API void builtin_extract_vec4l_int32(vec4l& a, int32 i, int64* ret);
WP_API void builtin_extract_vec2b_int32(vec2b& a, int32 i, int8* ret);
WP_API void builtin_extract_vec3b_int32(vec3b& a, int32 i, int8* ret);
WP_API void builtin_extract_vec4b_int32(vec4b& a, int32 i, int8* ret);
WP_API void builtin_extract_vec2us_int32(vec2us& a, int32 i, uint16* ret);
WP_API void builtin_extract_vec3us_int32(vec3us& a, int32 i, uint16* ret);
WP_API void builtin_extract_vec4us_int32(vec4us& a, int32 i, uint16* ret);
WP_API void builtin_extract_vec2ui_int32(vec2ui& a, int32 i, uint32* ret);
WP_API void builtin_extract_vec3ui_int32(vec3ui& a, int32 i, uint32* ret);
WP_API void builtin_extract_vec4ui_int32(vec4ui& a, int32 i, uint32* ret);
WP_API void builtin_extract_vec2ul_int32(vec2ul& a, int32 i, uint64* ret);
WP_API void builtin_extract_vec3ul_int32(vec3ul& a, int32 i, uint64* ret);
WP_API void builtin_extract_vec4ul_int32(vec4ul& a, int32 i, uint64* ret);
WP_API void builtin_extract_vec2ub_int32(vec2ub& a, int32 i, uint8* ret);
WP_API void builtin_extract_vec3ub_int32(vec3ub& a, int32 i, uint8* ret);
WP_API void builtin_extract_vec4ub_int32(vec4ub& a, int32 i, uint8* ret);
WP_API void builtin_extract_quath_int32(quath& a, int32 i, float16* ret);
WP_API void builtin_extract_quatf_int32(quatf& a, int32 i, float32* ret);
WP_API void builtin_extract_quatd_int32(quatd& a, int32 i, float64* ret);
WP_API void builtin_extract_mat22h_int32(mat22h& a, int32 i, vec2h* ret);
WP_API void builtin_extract_mat33h_int32(mat33h& a, int32 i, vec3h* ret);
WP_API void builtin_extract_mat44h_int32(mat44h& a, int32 i, vec4h* ret);
WP_API void builtin_extract_spatial_matrixh_int32(spatial_matrixh& a, int32 i, spatial_vectorh* ret);
WP_API void builtin_extract_mat22f_int32(mat22f& a, int32 i, vec2f* ret);
WP_API void builtin_extract_mat33f_int32(mat33f& a, int32 i, vec3f* ret);
WP_API void builtin_extract_mat44f_int32(mat44f& a, int32 i, vec4f* ret);
WP_API void builtin_extract_spatial_matrixf_int32(spatial_matrixf& a, int32 i, spatial_vectorf* ret);
WP_API void builtin_extract_mat22d_int32(mat22d& a, int32 i, vec2d* ret);
WP_API void builtin_extract_mat33d_int32(mat33d& a, int32 i, vec3d* ret);
WP_API void builtin_extract_mat44d_int32(mat44d& a, int32 i, vec4d* ret);
WP_API void builtin_extract_spatial_matrixd_int32(spatial_matrixd& a, int32 i, spatial_vectord* ret);
WP_API void builtin_extract_mat22h_int32_int32(mat22h& a, int32 i, int32 j, float16* ret);
WP_API void builtin_extract_mat33h_int32_int32(mat33h& a, int32 i, int32 j, float16* ret);
WP_API void builtin_extract_mat44h_int32_int32(mat44h& a, int32 i, int32 j, float16* ret);
WP_API void builtin_extract_spatial_matrixh_int32_int32(spatial_matrixh& a, int32 i, int32 j, float16* ret);
WP_API void builtin_extract_mat22f_int32_int32(mat22f& a, int32 i, int32 j, float32* ret);
WP_API void builtin_extract_mat33f_int32_int32(mat33f& a, int32 i, int32 j, float32* ret);
WP_API void builtin_extract_mat44f_int32_int32(mat44f& a, int32 i, int32 j, float32* ret);
WP_API void builtin_extract_spatial_matrixf_int32_int32(spatial_matrixf& a, int32 i, int32 j, float32* ret);
WP_API void builtin_extract_mat22d_int32_int32(mat22d& a, int32 i, int32 j, float64* ret);
WP_API void builtin_extract_mat33d_int32_int32(mat33d& a, int32 i, int32 j, float64* ret);
WP_API void builtin_extract_mat44d_int32_int32(mat44d& a, int32 i, int32 j, float64* ret);
WP_API void builtin_extract_spatial_matrixd_int32_int32(spatial_matrixd& a, int32 i, int32 j, float64* ret);
WP_API void builtin_extract_transformh_int32(transformh& a, int32 i, float16* ret);
WP_API void builtin_extract_transformf_int32(transformf& a, int32 i, float32* ret);
WP_API void builtin_extract_transformd_int32(transformd& a, int32 i, float64* ret);
WP_API void builtin_extract_shape_t_int32(shape_t s, int32 i, int* ret);
WP_API void builtin_lerp_float16_float16_float16(float16 a, float16 b, float16 t, float16* ret);
WP_API void builtin_lerp_float32_float32_float32(float32 a, float32 b, float32 t, float32* ret);
WP_API void builtin_lerp_float64_float64_float64(float64 a, float64 b, float64 t, float64* ret);
WP_API void builtin_lerp_vec2h_vec2h_float16(vec2h& a, vec2h& b, float16 t, vec2h* ret);
WP_API void builtin_lerp_vec3h_vec3h_float16(vec3h& a, vec3h& b, float16 t, vec3h* ret);
WP_API void builtin_lerp_vec4h_vec4h_float16(vec4h& a, vec4h& b, float16 t, vec4h* ret);
WP_API void builtin_lerp_spatial_vectorh_spatial_vectorh_float16(spatial_vectorh& a, spatial_vectorh& b, float16 t, spatial_vectorh* ret);
WP_API void builtin_lerp_vec2f_vec2f_float32(vec2f& a, vec2f& b, float32 t, vec2f* ret);
WP_API void builtin_lerp_vec3f_vec3f_float32(vec3f& a, vec3f& b, float32 t, vec3f* ret);
WP_API void builtin_lerp_vec4f_vec4f_float32(vec4f& a, vec4f& b, float32 t, vec4f* ret);
WP_API void builtin_lerp_spatial_vectorf_spatial_vectorf_float32(spatial_vectorf& a, spatial_vectorf& b, float32 t, spatial_vectorf* ret);
WP_API void builtin_lerp_vec2d_vec2d_float64(vec2d& a, vec2d& b, float64 t, vec2d* ret);
WP_API void builtin_lerp_vec3d_vec3d_float64(vec3d& a, vec3d& b, float64 t, vec3d* ret);
WP_API void builtin_lerp_vec4d_vec4d_float64(vec4d& a, vec4d& b, float64 t, vec4d* ret);
WP_API void builtin_lerp_spatial_vectord_spatial_vectord_float64(spatial_vectord& a, spatial_vectord& b, float64 t, spatial_vectord* ret);
WP_API void builtin_lerp_mat22h_mat22h_float16(mat22h& a, mat22h& b, float16 t, mat22h* ret);
WP_API void builtin_lerp_mat33h_mat33h_float16(mat33h& a, mat33h& b, float16 t, mat33h* ret);
WP_API void builtin_lerp_mat44h_mat44h_float16(mat44h& a, mat44h& b, float16 t, mat44h* ret);
WP_API void builtin_lerp_spatial_matrixh_spatial_matrixh_float16(spatial_matrixh& a, spatial_matrixh& b, float16 t, spatial_matrixh* ret);
WP_API void builtin_lerp_mat22f_mat22f_float32(mat22f& a, mat22f& b, float32 t, mat22f* ret);
WP_API void builtin_lerp_mat33f_mat33f_float32(mat33f& a, mat33f& b, float32 t, mat33f* ret);
WP_API void builtin_lerp_mat44f_mat44f_float32(mat44f& a, mat44f& b, float32 t, mat44f* ret);
WP_API void builtin_lerp_spatial_matrixf_spatial_matrixf_float32(spatial_matrixf& a, spatial_matrixf& b, float32 t, spatial_matrixf* ret);
WP_API void builtin_lerp_mat22d_mat22d_float64(mat22d& a, mat22d& b, float64 t, mat22d* ret);
WP_API void builtin_lerp_mat33d_mat33d_float64(mat33d& a, mat33d& b, float64 t, mat33d* ret);
WP_API void builtin_lerp_mat44d_mat44d_float64(mat44d& a, mat44d& b, float64 t, mat44d* ret);
WP_API void builtin_lerp_spatial_matrixd_spatial_matrixd_float64(spatial_matrixd& a, spatial_matrixd& b, float64 t, spatial_matrixd* ret);
WP_API void builtin_lerp_quath_quath_float16(quath& a, quath& b, float16 t, quath* ret);
WP_API void builtin_lerp_quatf_quatf_float32(quatf& a, quatf& b, float32 t, quatf* ret);
WP_API void builtin_lerp_quatd_quatd_float64(quatd& a, quatd& b, float64 t, quatd* ret);
WP_API void builtin_lerp_transformh_transformh_float16(transformh& a, transformh& b, float16 t, transformh* ret);
WP_API void builtin_lerp_transformf_transformf_float32(transformf& a, transformf& b, float32 t, transformf* ret);
WP_API void builtin_lerp_transformd_transformd_float64(transformd& a, transformd& b, float64 t, transformd* ret);
WP_API void builtin_smoothstep_float16_float16_float16(float16 a, float16 b, float16 x, float16* ret);
WP_API void builtin_smoothstep_float32_float32_float32(float32 a, float32 b, float32 x, float32* ret);
WP_API void builtin_smoothstep_float64_float64_float64(float64 a, float64 b, float64 x, float64* ret);
WP_API void builtin_add_float16_float16(float16 a, float16 b, float16* ret);
WP_API void builtin_add_float32_float32(float32 a, float32 b, float32* ret);
WP_API void builtin_add_float64_float64(float64 a, float64 b, float64* ret);
WP_API void builtin_add_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_add_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_add_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_add_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_add_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_add_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_add_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_add_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_add_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret);
WP_API void builtin_add_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret);
WP_API void builtin_add_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret);
WP_API void builtin_add_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_add_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret);
WP_API void builtin_add_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret);
WP_API void builtin_add_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret);
WP_API void builtin_add_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_add_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret);
WP_API void builtin_add_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret);
WP_API void builtin_add_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret);
WP_API void builtin_add_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_add_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret);
WP_API void builtin_add_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret);
WP_API void builtin_add_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret);
WP_API void builtin_add_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret);
WP_API void builtin_add_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret);
WP_API void builtin_add_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret);
WP_API void builtin_add_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret);
WP_API void builtin_add_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret);
WP_API void builtin_add_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret);
WP_API void builtin_add_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret);
WP_API void builtin_add_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret);
WP_API void builtin_add_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret);
WP_API void builtin_add_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret);
WP_API void builtin_add_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret);
WP_API void builtin_add_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret);
WP_API void builtin_add_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret);
WP_API void builtin_add_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret);
WP_API void builtin_add_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret);
WP_API void builtin_add_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret);
WP_API void builtin_add_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret);
WP_API void builtin_add_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret);
WP_API void builtin_add_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret);
WP_API void builtin_add_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret);
WP_API void builtin_add_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret);
WP_API void builtin_add_quath_quath(quath& a, quath& b, quath* ret);
WP_API void builtin_add_quatf_quatf(quatf& a, quatf& b, quatf* ret);
WP_API void builtin_add_quatd_quatd(quatd& a, quatd& b, quatd* ret);
WP_API void builtin_add_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret);
WP_API void builtin_add_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret);
WP_API void builtin_add_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret);
WP_API void builtin_add_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret);
WP_API void builtin_add_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret);
WP_API void builtin_add_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret);
WP_API void builtin_add_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret);
WP_API void builtin_add_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret);
WP_API void builtin_add_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret);
WP_API void builtin_add_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret);
WP_API void builtin_add_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret);
WP_API void builtin_add_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret);
WP_API void builtin_add_transformh_transformh(transformh& a, transformh& b, transformh* ret);
WP_API void builtin_add_transformf_transformf(transformf& a, transformf& b, transformf* ret);
WP_API void builtin_add_transformd_transformd(transformd& a, transformd& b, transformd* ret);
WP_API void builtin_sub_float16_float16(float16 a, float16 b, float16* ret);
WP_API void builtin_sub_float32_float32(float32 a, float32 b, float32* ret);
WP_API void builtin_sub_float64_float64(float64 a, float64 b, float64* ret);
WP_API void builtin_sub_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_sub_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_sub_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_sub_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_sub_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_sub_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_sub_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_sub_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_sub_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret);
WP_API void builtin_sub_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret);
WP_API void builtin_sub_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret);
WP_API void builtin_sub_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_sub_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret);
WP_API void builtin_sub_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret);
WP_API void builtin_sub_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret);
WP_API void builtin_sub_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_sub_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret);
WP_API void builtin_sub_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret);
WP_API void builtin_sub_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret);
WP_API void builtin_sub_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_sub_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret);
WP_API void builtin_sub_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret);
WP_API void builtin_sub_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret);
WP_API void builtin_sub_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret);
WP_API void builtin_sub_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret);
WP_API void builtin_sub_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret);
WP_API void builtin_sub_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret);
WP_API void builtin_sub_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret);
WP_API void builtin_sub_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret);
WP_API void builtin_sub_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret);
WP_API void builtin_sub_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret);
WP_API void builtin_sub_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret);
WP_API void builtin_sub_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret);
WP_API void builtin_sub_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret);
WP_API void builtin_sub_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret);
WP_API void builtin_sub_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret);
WP_API void builtin_sub_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret);
WP_API void builtin_sub_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret);
WP_API void builtin_sub_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret);
WP_API void builtin_sub_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret);
WP_API void builtin_sub_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret);
WP_API void builtin_sub_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret);
WP_API void builtin_sub_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret);
WP_API void builtin_sub_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret);
WP_API void builtin_sub_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret);
WP_API void builtin_sub_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret);
WP_API void builtin_sub_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret);
WP_API void builtin_sub_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret);
WP_API void builtin_sub_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret);
WP_API void builtin_sub_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret);
WP_API void builtin_sub_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret);
WP_API void builtin_sub_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret);
WP_API void builtin_sub_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret);
WP_API void builtin_sub_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret);
WP_API void builtin_sub_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret);
WP_API void builtin_sub_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret);
WP_API void builtin_sub_quath_quath(quath& a, quath& b, quath* ret);
WP_API void builtin_sub_quatf_quatf(quatf& a, quatf& b, quatf* ret);
WP_API void builtin_sub_quatd_quatd(quatd& a, quatd& b, quatd* ret);
WP_API void builtin_sub_transformh_transformh(transformh& a, transformh& b, transformh* ret);
WP_API void builtin_sub_transformf_transformf(transformf& a, transformf& b, transformf* ret);
WP_API void builtin_sub_transformd_transformd(transformd& a, transformd& b, transformd* ret);
WP_API void builtin_bit_and_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_bit_and_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_bit_and_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_bit_and_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_bit_and_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_bit_and_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_bit_and_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_bit_and_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_bit_or_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_bit_or_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_bit_or_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_bit_or_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_bit_or_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_bit_or_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_bit_or_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_bit_or_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_bit_xor_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_bit_xor_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_bit_xor_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_bit_xor_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_bit_xor_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_bit_xor_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_bit_xor_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_bit_xor_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_lshift_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_lshift_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_lshift_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_lshift_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_lshift_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_lshift_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_lshift_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_lshift_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_rshift_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_rshift_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_rshift_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_rshift_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_rshift_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_rshift_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_rshift_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_rshift_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_invert_int16(int16 a, int16* ret);
WP_API void builtin_invert_int32(int32 a, int32* ret);
WP_API void builtin_invert_int64(int64 a, int64* ret);
WP_API void builtin_invert_int8(int8 a, int8* ret);
WP_API void builtin_invert_uint16(uint16 a, uint16* ret);
WP_API void builtin_invert_uint32(uint32 a, uint32* ret);
WP_API void builtin_invert_uint64(uint64 a, uint64* ret);
WP_API void builtin_invert_uint8(uint8 a, uint8* ret);
WP_API void builtin_mul_float16_float16(float16 a, float16 b, float16* ret);
WP_API void builtin_mul_float32_float32(float32 a, float32 b, float32* ret);
WP_API void builtin_mul_float64_float64(float64 a, float64 b, float64* ret);
WP_API void builtin_mul_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_mul_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_mul_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_mul_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_mul_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_mul_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_mul_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_mul_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_mul_vec2h_float16(vec2h& a, float16 b, vec2h* ret);
WP_API void builtin_mul_vec3h_float16(vec3h& a, float16 b, vec3h* ret);
WP_API void builtin_mul_vec4h_float16(vec4h& a, float16 b, vec4h* ret);
WP_API void builtin_mul_spatial_vectorh_float16(spatial_vectorh& a, float16 b, spatial_vectorh* ret);
WP_API void builtin_mul_vec2f_float32(vec2f& a, float32 b, vec2f* ret);
WP_API void builtin_mul_vec3f_float32(vec3f& a, float32 b, vec3f* ret);
WP_API void builtin_mul_vec4f_float32(vec4f& a, float32 b, vec4f* ret);
WP_API void builtin_mul_spatial_vectorf_float32(spatial_vectorf& a, float32 b, spatial_vectorf* ret);
WP_API void builtin_mul_vec2d_float64(vec2d& a, float64 b, vec2d* ret);
WP_API void builtin_mul_vec3d_float64(vec3d& a, float64 b, vec3d* ret);
WP_API void builtin_mul_vec4d_float64(vec4d& a, float64 b, vec4d* ret);
WP_API void builtin_mul_spatial_vectord_float64(spatial_vectord& a, float64 b, spatial_vectord* ret);
WP_API void builtin_mul_vec2s_int16(vec2s& a, int16 b, vec2s* ret);
WP_API void builtin_mul_vec3s_int16(vec3s& a, int16 b, vec3s* ret);
WP_API void builtin_mul_vec4s_int16(vec4s& a, int16 b, vec4s* ret);
WP_API void builtin_mul_vec2i_int32(vec2i& a, int32 b, vec2i* ret);
WP_API void builtin_mul_vec3i_int32(vec3i& a, int32 b, vec3i* ret);
WP_API void builtin_mul_vec4i_int32(vec4i& a, int32 b, vec4i* ret);
WP_API void builtin_mul_vec2l_int64(vec2l& a, int64 b, vec2l* ret);
WP_API void builtin_mul_vec3l_int64(vec3l& a, int64 b, vec3l* ret);
WP_API void builtin_mul_vec4l_int64(vec4l& a, int64 b, vec4l* ret);
WP_API void builtin_mul_vec2b_int8(vec2b& a, int8 b, vec2b* ret);
WP_API void builtin_mul_vec3b_int8(vec3b& a, int8 b, vec3b* ret);
WP_API void builtin_mul_vec4b_int8(vec4b& a, int8 b, vec4b* ret);
WP_API void builtin_mul_vec2us_uint16(vec2us& a, uint16 b, vec2us* ret);
WP_API void builtin_mul_vec3us_uint16(vec3us& a, uint16 b, vec3us* ret);
WP_API void builtin_mul_vec4us_uint16(vec4us& a, uint16 b, vec4us* ret);
WP_API void builtin_mul_vec2ui_uint32(vec2ui& a, uint32 b, vec2ui* ret);
WP_API void builtin_mul_vec3ui_uint32(vec3ui& a, uint32 b, vec3ui* ret);
WP_API void builtin_mul_vec4ui_uint32(vec4ui& a, uint32 b, vec4ui* ret);
WP_API void builtin_mul_vec2ul_uint64(vec2ul& a, uint64 b, vec2ul* ret);
WP_API void builtin_mul_vec3ul_uint64(vec3ul& a, uint64 b, vec3ul* ret);
WP_API void builtin_mul_vec4ul_uint64(vec4ul& a, uint64 b, vec4ul* ret);
WP_API void builtin_mul_vec2ub_uint8(vec2ub& a, uint8 b, vec2ub* ret);
WP_API void builtin_mul_vec3ub_uint8(vec3ub& a, uint8 b, vec3ub* ret);
WP_API void builtin_mul_vec4ub_uint8(vec4ub& a, uint8 b, vec4ub* ret);
WP_API void builtin_mul_float16_vec2h(float16 a, vec2h& b, vec2h* ret);
WP_API void builtin_mul_float16_vec3h(float16 a, vec3h& b, vec3h* ret);
WP_API void builtin_mul_float16_vec4h(float16 a, vec4h& b, vec4h* ret);
WP_API void builtin_mul_float16_spatial_vectorh(float16 a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_mul_float32_vec2f(float32 a, vec2f& b, vec2f* ret);
WP_API void builtin_mul_float32_vec3f(float32 a, vec3f& b, vec3f* ret);
WP_API void builtin_mul_float32_vec4f(float32 a, vec4f& b, vec4f* ret);
WP_API void builtin_mul_float32_spatial_vectorf(float32 a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_mul_float64_vec2d(float64 a, vec2d& b, vec2d* ret);
WP_API void builtin_mul_float64_vec3d(float64 a, vec3d& b, vec3d* ret);
WP_API void builtin_mul_float64_vec4d(float64 a, vec4d& b, vec4d* ret);
WP_API void builtin_mul_float64_spatial_vectord(float64 a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_mul_int16_vec2s(int16 a, vec2s& b, vec2s* ret);
WP_API void builtin_mul_int16_vec3s(int16 a, vec3s& b, vec3s* ret);
WP_API void builtin_mul_int16_vec4s(int16 a, vec4s& b, vec4s* ret);
WP_API void builtin_mul_int32_vec2i(int32 a, vec2i& b, vec2i* ret);
WP_API void builtin_mul_int32_vec3i(int32 a, vec3i& b, vec3i* ret);
WP_API void builtin_mul_int32_vec4i(int32 a, vec4i& b, vec4i* ret);
WP_API void builtin_mul_int64_vec2l(int64 a, vec2l& b, vec2l* ret);
WP_API void builtin_mul_int64_vec3l(int64 a, vec3l& b, vec3l* ret);
WP_API void builtin_mul_int64_vec4l(int64 a, vec4l& b, vec4l* ret);
WP_API void builtin_mul_int8_vec2b(int8 a, vec2b& b, vec2b* ret);
WP_API void builtin_mul_int8_vec3b(int8 a, vec3b& b, vec3b* ret);
WP_API void builtin_mul_int8_vec4b(int8 a, vec4b& b, vec4b* ret);
WP_API void builtin_mul_uint16_vec2us(uint16 a, vec2us& b, vec2us* ret);
WP_API void builtin_mul_uint16_vec3us(uint16 a, vec3us& b, vec3us* ret);
WP_API void builtin_mul_uint16_vec4us(uint16 a, vec4us& b, vec4us* ret);
WP_API void builtin_mul_uint32_vec2ui(uint32 a, vec2ui& b, vec2ui* ret);
WP_API void builtin_mul_uint32_vec3ui(uint32 a, vec3ui& b, vec3ui* ret);
WP_API void builtin_mul_uint32_vec4ui(uint32 a, vec4ui& b, vec4ui* ret);
WP_API void builtin_mul_uint64_vec2ul(uint64 a, vec2ul& b, vec2ul* ret);
WP_API void builtin_mul_uint64_vec3ul(uint64 a, vec3ul& b, vec3ul* ret);
WP_API void builtin_mul_uint64_vec4ul(uint64 a, vec4ul& b, vec4ul* ret);
WP_API void builtin_mul_uint8_vec2ub(uint8 a, vec2ub& b, vec2ub* ret);
WP_API void builtin_mul_uint8_vec3ub(uint8 a, vec3ub& b, vec3ub* ret);
WP_API void builtin_mul_uint8_vec4ub(uint8 a, vec4ub& b, vec4ub* ret);
WP_API void builtin_mul_quath_float16(quath& a, float16 b, quath* ret);
WP_API void builtin_mul_quatf_float32(quatf& a, float32 b, quatf* ret);
WP_API void builtin_mul_quatd_float64(quatd& a, float64 b, quatd* ret);
WP_API void builtin_mul_float16_quath(float16 a, quath& b, quath* ret);
WP_API void builtin_mul_float32_quatf(float32 a, quatf& b, quatf* ret);
WP_API void builtin_mul_float64_quatd(float64 a, quatd& b, quatd* ret);
WP_API void builtin_mul_quath_quath(quath& a, quath& b, quath* ret);
WP_API void builtin_mul_quatf_quatf(quatf& a, quatf& b, quatf* ret);
WP_API void builtin_mul_quatd_quatd(quatd& a, quatd& b, quatd* ret);
WP_API void builtin_mul_float16_mat22h(float16 a, mat22h& b, mat22h* ret);
WP_API void builtin_mul_float16_mat33h(float16 a, mat33h& b, mat33h* ret);
WP_API void builtin_mul_float16_mat44h(float16 a, mat44h& b, mat44h* ret);
WP_API void builtin_mul_float16_spatial_matrixh(float16 a, spatial_matrixh& b, spatial_matrixh* ret);
WP_API void builtin_mul_float32_mat22f(float32 a, mat22f& b, mat22f* ret);
WP_API void builtin_mul_float32_mat33f(float32 a, mat33f& b, mat33f* ret);
WP_API void builtin_mul_float32_mat44f(float32 a, mat44f& b, mat44f* ret);
WP_API void builtin_mul_float32_spatial_matrixf(float32 a, spatial_matrixf& b, spatial_matrixf* ret);
WP_API void builtin_mul_float64_mat22d(float64 a, mat22d& b, mat22d* ret);
WP_API void builtin_mul_float64_mat33d(float64 a, mat33d& b, mat33d* ret);
WP_API void builtin_mul_float64_mat44d(float64 a, mat44d& b, mat44d* ret);
WP_API void builtin_mul_float64_spatial_matrixd(float64 a, spatial_matrixd& b, spatial_matrixd* ret);
WP_API void builtin_mul_mat22h_float16(mat22h& a, float16 b, mat22h* ret);
WP_API void builtin_mul_mat33h_float16(mat33h& a, float16 b, mat33h* ret);
WP_API void builtin_mul_mat44h_float16(mat44h& a, float16 b, mat44h* ret);
WP_API void builtin_mul_spatial_matrixh_float16(spatial_matrixh& a, float16 b, spatial_matrixh* ret);
WP_API void builtin_mul_mat22f_float32(mat22f& a, float32 b, mat22f* ret);
WP_API void builtin_mul_mat33f_float32(mat33f& a, float32 b, mat33f* ret);
WP_API void builtin_mul_mat44f_float32(mat44f& a, float32 b, mat44f* ret);
WP_API void builtin_mul_spatial_matrixf_float32(spatial_matrixf& a, float32 b, spatial_matrixf* ret);
WP_API void builtin_mul_mat22d_float64(mat22d& a, float64 b, mat22d* ret);
WP_API void builtin_mul_mat33d_float64(mat33d& a, float64 b, mat33d* ret);
WP_API void builtin_mul_mat44d_float64(mat44d& a, float64 b, mat44d* ret);
WP_API void builtin_mul_spatial_matrixd_float64(spatial_matrixd& a, float64 b, spatial_matrixd* ret);
WP_API void builtin_mul_mat22h_vec2h(mat22h& a, vec2h& b, vec2h* ret);
WP_API void builtin_mul_mat33h_vec3h(mat33h& a, vec3h& b, vec3h* ret);
WP_API void builtin_mul_mat44h_vec4h(mat44h& a, vec4h& b, vec4h* ret);
WP_API void builtin_mul_spatial_matrixh_spatial_vectorh(spatial_matrixh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_mul_mat22f_vec2f(mat22f& a, vec2f& b, vec2f* ret);
WP_API void builtin_mul_mat33f_vec3f(mat33f& a, vec3f& b, vec3f* ret);
WP_API void builtin_mul_mat44f_vec4f(mat44f& a, vec4f& b, vec4f* ret);
WP_API void builtin_mul_spatial_matrixf_spatial_vectorf(spatial_matrixf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_mul_mat22d_vec2d(mat22d& a, vec2d& b, vec2d* ret);
WP_API void builtin_mul_mat33d_vec3d(mat33d& a, vec3d& b, vec3d* ret);
WP_API void builtin_mul_mat44d_vec4d(mat44d& a, vec4d& b, vec4d* ret);
WP_API void builtin_mul_spatial_matrixd_spatial_vectord(spatial_matrixd& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_mul_vec2h_mat22h(vec2h& a, mat22h& b, vec2h* ret);
WP_API void builtin_mul_vec3h_mat33h(vec3h& a, mat33h& b, vec3h* ret);
WP_API void builtin_mul_vec4h_mat44h(vec4h& a, mat44h& b, vec4h* ret);
WP_API void builtin_mul_spatial_vectorh_spatial_matrixh(spatial_vectorh& a, spatial_matrixh& b, spatial_vectorh* ret);
WP_API void builtin_mul_vec2f_mat22f(vec2f& a, mat22f& b, vec2f* ret);
WP_API void builtin_mul_vec3f_mat33f(vec3f& a, mat33f& b, vec3f* ret);
WP_API void builtin_mul_vec4f_mat44f(vec4f& a, mat44f& b, vec4f* ret);
WP_API void builtin_mul_spatial_vectorf_spatial_matrixf(spatial_vectorf& a, spatial_matrixf& b, spatial_vectorf* ret);
WP_API void builtin_mul_vec2d_mat22d(vec2d& a, mat22d& b, vec2d* ret);
WP_API void builtin_mul_vec3d_mat33d(vec3d& a, mat33d& b, vec3d* ret);
WP_API void builtin_mul_vec4d_mat44d(vec4d& a, mat44d& b, vec4d* ret);
WP_API void builtin_mul_spatial_vectord_spatial_matrixd(spatial_vectord& a, spatial_matrixd& b, spatial_vectord* ret);
WP_API void builtin_mul_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret);
WP_API void builtin_mul_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret);
WP_API void builtin_mul_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret);
WP_API void builtin_mul_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret);
WP_API void builtin_mul_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret);
WP_API void builtin_mul_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret);
WP_API void builtin_mul_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret);
WP_API void builtin_mul_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret);
WP_API void builtin_mul_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret);
WP_API void builtin_mul_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret);
WP_API void builtin_mul_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret);
WP_API void builtin_mul_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret);
WP_API void builtin_mul_transformh_transformh(transformh& a, transformh& b, transformh* ret);
WP_API void builtin_mul_transformf_transformf(transformf& a, transformf& b, transformf* ret);
WP_API void builtin_mul_transformd_transformd(transformd& a, transformd& b, transformd* ret);
WP_API void builtin_mul_float16_transformh(float16 a, transformh& b, transformh* ret);
WP_API void builtin_mul_float32_transformf(float32 a, transformf& b, transformf* ret);
WP_API void builtin_mul_float64_transformd(float64 a, transformd& b, transformd* ret);
WP_API void builtin_mul_transformh_float16(transformh& a, float16 b, transformh* ret);
WP_API void builtin_mul_transformf_float32(transformf& a, float32 b, transformf* ret);
WP_API void builtin_mul_transformd_float64(transformd& a, float64 b, transformd* ret);
WP_API void builtin_mod_float16_float16(float16 a, float16 b, float16* ret);
WP_API void builtin_mod_float32_float32(float32 a, float32 b, float32* ret);
WP_API void builtin_mod_float64_float64(float64 a, float64 b, float64* ret);
WP_API void builtin_mod_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_mod_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_mod_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_mod_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_mod_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_mod_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_mod_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_mod_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_mod_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret);
WP_API void builtin_mod_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret);
WP_API void builtin_mod_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret);
WP_API void builtin_mod_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_mod_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret);
WP_API void builtin_mod_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret);
WP_API void builtin_mod_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret);
WP_API void builtin_mod_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_mod_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret);
WP_API void builtin_mod_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret);
WP_API void builtin_mod_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret);
WP_API void builtin_mod_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_mod_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret);
WP_API void builtin_mod_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret);
WP_API void builtin_mod_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret);
WP_API void builtin_mod_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret);
WP_API void builtin_mod_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret);
WP_API void builtin_mod_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret);
WP_API void builtin_mod_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret);
WP_API void builtin_mod_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret);
WP_API void builtin_mod_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret);
WP_API void builtin_mod_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret);
WP_API void builtin_mod_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret);
WP_API void builtin_mod_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret);
WP_API void builtin_mod_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret);
WP_API void builtin_mod_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret);
WP_API void builtin_mod_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret);
WP_API void builtin_mod_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret);
WP_API void builtin_mod_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret);
WP_API void builtin_mod_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret);
WP_API void builtin_mod_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret);
WP_API void builtin_mod_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret);
WP_API void builtin_mod_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret);
WP_API void builtin_mod_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret);
WP_API void builtin_mod_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret);
WP_API void builtin_mod_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret);
WP_API void builtin_div_float16_float16(float16 a, float16 b, float16* ret);
WP_API void builtin_div_float32_float32(float32 a, float32 b, float32* ret);
WP_API void builtin_div_float64_float64(float64 a, float64 b, float64* ret);
WP_API void builtin_div_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_div_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_div_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_div_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_div_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_div_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_div_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_div_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_div_vec2h_float16(vec2h& a, float16 b, vec2h* ret);
WP_API void builtin_div_vec3h_float16(vec3h& a, float16 b, vec3h* ret);
WP_API void builtin_div_vec4h_float16(vec4h& a, float16 b, vec4h* ret);
WP_API void builtin_div_spatial_vectorh_float16(spatial_vectorh& a, float16 b, spatial_vectorh* ret);
WP_API void builtin_div_vec2f_float32(vec2f& a, float32 b, vec2f* ret);
WP_API void builtin_div_vec3f_float32(vec3f& a, float32 b, vec3f* ret);
WP_API void builtin_div_vec4f_float32(vec4f& a, float32 b, vec4f* ret);
WP_API void builtin_div_spatial_vectorf_float32(spatial_vectorf& a, float32 b, spatial_vectorf* ret);
WP_API void builtin_div_vec2d_float64(vec2d& a, float64 b, vec2d* ret);
WP_API void builtin_div_vec3d_float64(vec3d& a, float64 b, vec3d* ret);
WP_API void builtin_div_vec4d_float64(vec4d& a, float64 b, vec4d* ret);
WP_API void builtin_div_spatial_vectord_float64(spatial_vectord& a, float64 b, spatial_vectord* ret);
WP_API void builtin_div_vec2s_int16(vec2s& a, int16 b, vec2s* ret);
WP_API void builtin_div_vec3s_int16(vec3s& a, int16 b, vec3s* ret);
WP_API void builtin_div_vec4s_int16(vec4s& a, int16 b, vec4s* ret);
WP_API void builtin_div_vec2i_int32(vec2i& a, int32 b, vec2i* ret);
WP_API void builtin_div_vec3i_int32(vec3i& a, int32 b, vec3i* ret);
WP_API void builtin_div_vec4i_int32(vec4i& a, int32 b, vec4i* ret);
WP_API void builtin_div_vec2l_int64(vec2l& a, int64 b, vec2l* ret);
WP_API void builtin_div_vec3l_int64(vec3l& a, int64 b, vec3l* ret);
WP_API void builtin_div_vec4l_int64(vec4l& a, int64 b, vec4l* ret);
WP_API void builtin_div_vec2b_int8(vec2b& a, int8 b, vec2b* ret);
WP_API void builtin_div_vec3b_int8(vec3b& a, int8 b, vec3b* ret);
WP_API void builtin_div_vec4b_int8(vec4b& a, int8 b, vec4b* ret);
WP_API void builtin_div_vec2us_uint16(vec2us& a, uint16 b, vec2us* ret);
WP_API void builtin_div_vec3us_uint16(vec3us& a, uint16 b, vec3us* ret);
WP_API void builtin_div_vec4us_uint16(vec4us& a, uint16 b, vec4us* ret);
WP_API void builtin_div_vec2ui_uint32(vec2ui& a, uint32 b, vec2ui* ret);
WP_API void builtin_div_vec3ui_uint32(vec3ui& a, uint32 b, vec3ui* ret);
WP_API void builtin_div_vec4ui_uint32(vec4ui& a, uint32 b, vec4ui* ret);
WP_API void builtin_div_vec2ul_uint64(vec2ul& a, uint64 b, vec2ul* ret);
WP_API void builtin_div_vec3ul_uint64(vec3ul& a, uint64 b, vec3ul* ret);
WP_API void builtin_div_vec4ul_uint64(vec4ul& a, uint64 b, vec4ul* ret);
WP_API void builtin_div_vec2ub_uint8(vec2ub& a, uint8 b, vec2ub* ret);
WP_API void builtin_div_vec3ub_uint8(vec3ub& a, uint8 b, vec3ub* ret);
WP_API void builtin_div_vec4ub_uint8(vec4ub& a, uint8 b, vec4ub* ret);
WP_API void builtin_div_float16_vec2h(float16 a, vec2h& b, vec2h* ret);
WP_API void builtin_div_float16_vec3h(float16 a, vec3h& b, vec3h* ret);
WP_API void builtin_div_float16_vec4h(float16 a, vec4h& b, vec4h* ret);
WP_API void builtin_div_float16_spatial_vectorh(float16 a, spatial_vectorh& b, spatial_vectorh* ret);
WP_API void builtin_div_float32_vec2f(float32 a, vec2f& b, vec2f* ret);
WP_API void builtin_div_float32_vec3f(float32 a, vec3f& b, vec3f* ret);
WP_API void builtin_div_float32_vec4f(float32 a, vec4f& b, vec4f* ret);
WP_API void builtin_div_float32_spatial_vectorf(float32 a, spatial_vectorf& b, spatial_vectorf* ret);
WP_API void builtin_div_float64_vec2d(float64 a, vec2d& b, vec2d* ret);
WP_API void builtin_div_float64_vec3d(float64 a, vec3d& b, vec3d* ret);
WP_API void builtin_div_float64_vec4d(float64 a, vec4d& b, vec4d* ret);
WP_API void builtin_div_float64_spatial_vectord(float64 a, spatial_vectord& b, spatial_vectord* ret);
WP_API void builtin_div_int16_vec2s(int16 a, vec2s& b, vec2s* ret);
WP_API void builtin_div_int16_vec3s(int16 a, vec3s& b, vec3s* ret);
WP_API void builtin_div_int16_vec4s(int16 a, vec4s& b, vec4s* ret);
WP_API void builtin_div_int32_vec2i(int32 a, vec2i& b, vec2i* ret);
WP_API void builtin_div_int32_vec3i(int32 a, vec3i& b, vec3i* ret);
WP_API void builtin_div_int32_vec4i(int32 a, vec4i& b, vec4i* ret);
WP_API void builtin_div_int64_vec2l(int64 a, vec2l& b, vec2l* ret);
WP_API void builtin_div_int64_vec3l(int64 a, vec3l& b, vec3l* ret);
WP_API void builtin_div_int64_vec4l(int64 a, vec4l& b, vec4l* ret);
WP_API void builtin_div_int8_vec2b(int8 a, vec2b& b, vec2b* ret);
WP_API void builtin_div_int8_vec3b(int8 a, vec3b& b, vec3b* ret);
WP_API void builtin_div_int8_vec4b(int8 a, vec4b& b, vec4b* ret);
WP_API void builtin_div_uint16_vec2us(uint16 a, vec2us& b, vec2us* ret);
WP_API void builtin_div_uint16_vec3us(uint16 a, vec3us& b, vec3us* ret);
WP_API void builtin_div_uint16_vec4us(uint16 a, vec4us& b, vec4us* ret);
WP_API void builtin_div_uint32_vec2ui(uint32 a, vec2ui& b, vec2ui* ret);
WP_API void builtin_div_uint32_vec3ui(uint32 a, vec3ui& b, vec3ui* ret);
WP_API void builtin_div_uint32_vec4ui(uint32 a, vec4ui& b, vec4ui* ret);
WP_API void builtin_div_uint64_vec2ul(uint64 a, vec2ul& b, vec2ul* ret);
WP_API void builtin_div_uint64_vec3ul(uint64 a, vec3ul& b, vec3ul* ret);
WP_API void builtin_div_uint64_vec4ul(uint64 a, vec4ul& b, vec4ul* ret);
WP_API void builtin_div_uint8_vec2ub(uint8 a, vec2ub& b, vec2ub* ret);
WP_API void builtin_div_uint8_vec3ub(uint8 a, vec3ub& b, vec3ub* ret);
WP_API void builtin_div_uint8_vec4ub(uint8 a, vec4ub& b, vec4ub* ret);
WP_API void builtin_div_mat22h_float16(mat22h& a, float16 b, mat22h* ret);
WP_API void builtin_div_mat33h_float16(mat33h& a, float16 b, mat33h* ret);
WP_API void builtin_div_mat44h_float16(mat44h& a, float16 b, mat44h* ret);
WP_API void builtin_div_spatial_matrixh_float16(spatial_matrixh& a, float16 b, spatial_matrixh* ret);
WP_API void builtin_div_mat22f_float32(mat22f& a, float32 b, mat22f* ret);
WP_API void builtin_div_mat33f_float32(mat33f& a, float32 b, mat33f* ret);
WP_API void builtin_div_mat44f_float32(mat44f& a, float32 b, mat44f* ret);
WP_API void builtin_div_spatial_matrixf_float32(spatial_matrixf& a, float32 b, spatial_matrixf* ret);
WP_API void builtin_div_mat22d_float64(mat22d& a, float64 b, mat22d* ret);
WP_API void builtin_div_mat33d_float64(mat33d& a, float64 b, mat33d* ret);
WP_API void builtin_div_mat44d_float64(mat44d& a, float64 b, mat44d* ret);
WP_API void builtin_div_spatial_matrixd_float64(spatial_matrixd& a, float64 b, spatial_matrixd* ret);
WP_API void builtin_div_float16_mat22h(float16 a, mat22h& b, mat22h* ret);
WP_API void builtin_div_float16_mat33h(float16 a, mat33h& b, mat33h* ret);
WP_API void builtin_div_float16_mat44h(float16 a, mat44h& b, mat44h* ret);
WP_API void builtin_div_float16_spatial_matrixh(float16 a, spatial_matrixh& b, spatial_matrixh* ret);
WP_API void builtin_div_float32_mat22f(float32 a, mat22f& b, mat22f* ret);
WP_API void builtin_div_float32_mat33f(float32 a, mat33f& b, mat33f* ret);
WP_API void builtin_div_float32_mat44f(float32 a, mat44f& b, mat44f* ret);
WP_API void builtin_div_float32_spatial_matrixf(float32 a, spatial_matrixf& b, spatial_matrixf* ret);
WP_API void builtin_div_float64_mat22d(float64 a, mat22d& b, mat22d* ret);
WP_API void builtin_div_float64_mat33d(float64 a, mat33d& b, mat33d* ret);
WP_API void builtin_div_float64_mat44d(float64 a, mat44d& b, mat44d* ret);
WP_API void builtin_div_float64_spatial_matrixd(float64 a, spatial_matrixd& b, spatial_matrixd* ret);
WP_API void builtin_div_quath_float16(quath& a, float16 b, quath* ret);
WP_API void builtin_div_quatf_float32(quatf& a, float32 b, quatf* ret);
WP_API void builtin_div_quatd_float64(quatd& a, float64 b, quatd* ret);
WP_API void builtin_div_float16_quath(float16 a, quath& b, quath* ret);
WP_API void builtin_div_float32_quatf(float32 a, quatf& b, quatf* ret);
WP_API void builtin_div_float64_quatd(float64 a, quatd& b, quatd* ret);
WP_API void builtin_floordiv_float16_float16(float16 a, float16 b, float16* ret);
WP_API void builtin_floordiv_float32_float32(float32 a, float32 b, float32* ret);
WP_API void builtin_floordiv_float64_float64(float64 a, float64 b, float64* ret);
WP_API void builtin_floordiv_int16_int16(int16 a, int16 b, int16* ret);
WP_API void builtin_floordiv_int32_int32(int32 a, int32 b, int32* ret);
WP_API void builtin_floordiv_int64_int64(int64 a, int64 b, int64* ret);
WP_API void builtin_floordiv_int8_int8(int8 a, int8 b, int8* ret);
WP_API void builtin_floordiv_uint16_uint16(uint16 a, uint16 b, uint16* ret);
WP_API void builtin_floordiv_uint32_uint32(uint32 a, uint32 b, uint32* ret);
WP_API void builtin_floordiv_uint64_uint64(uint64 a, uint64 b, uint64* ret);
WP_API void builtin_floordiv_uint8_uint8(uint8 a, uint8 b, uint8* ret);
WP_API void builtin_pos_float16(float16 x, float16* ret);
WP_API void builtin_pos_float32(float32 x, float32* ret);
WP_API void builtin_pos_float64(float64 x, float64* ret);
WP_API void builtin_pos_int16(int16 x, int16* ret);
WP_API void builtin_pos_int32(int32 x, int32* ret);
WP_API void builtin_pos_int64(int64 x, int64* ret);
WP_API void builtin_pos_int8(int8 x, int8* ret);
WP_API void builtin_pos_uint16(uint16 x, uint16* ret);
WP_API void builtin_pos_uint32(uint32 x, uint32* ret);
WP_API void builtin_pos_uint64(uint64 x, uint64* ret);
WP_API void builtin_pos_uint8(uint8 x, uint8* ret);
WP_API void builtin_pos_vec2h(vec2h& x, vec2h* ret);
WP_API void builtin_pos_vec3h(vec3h& x, vec3h* ret);
WP_API void builtin_pos_vec4h(vec4h& x, vec4h* ret);
WP_API void builtin_pos_spatial_vectorh(spatial_vectorh& x, spatial_vectorh* ret);
WP_API void builtin_pos_vec2f(vec2f& x, vec2f* ret);
WP_API void builtin_pos_vec3f(vec3f& x, vec3f* ret);
WP_API void builtin_pos_vec4f(vec4f& x, vec4f* ret);
WP_API void builtin_pos_spatial_vectorf(spatial_vectorf& x, spatial_vectorf* ret);
WP_API void builtin_pos_vec2d(vec2d& x, vec2d* ret);
WP_API void builtin_pos_vec3d(vec3d& x, vec3d* ret);
WP_API void builtin_pos_vec4d(vec4d& x, vec4d* ret);
WP_API void builtin_pos_spatial_vectord(spatial_vectord& x, spatial_vectord* ret);
WP_API void builtin_pos_vec2s(vec2s& x, vec2s* ret);
WP_API void builtin_pos_vec3s(vec3s& x, vec3s* ret);
WP_API void builtin_pos_vec4s(vec4s& x, vec4s* ret);
WP_API void builtin_pos_vec2i(vec2i& x, vec2i* ret);
WP_API void builtin_pos_vec3i(vec3i& x, vec3i* ret);
WP_API void builtin_pos_vec4i(vec4i& x, vec4i* ret);
WP_API void builtin_pos_vec2l(vec2l& x, vec2l* ret);
WP_API void builtin_pos_vec3l(vec3l& x, vec3l* ret);
WP_API void builtin_pos_vec4l(vec4l& x, vec4l* ret);
WP_API void builtin_pos_vec2b(vec2b& x, vec2b* ret);
WP_API void builtin_pos_vec3b(vec3b& x, vec3b* ret);
WP_API void builtin_pos_vec4b(vec4b& x, vec4b* ret);
WP_API void builtin_pos_vec2us(vec2us& x, vec2us* ret);
WP_API void builtin_pos_vec3us(vec3us& x, vec3us* ret);
WP_API void builtin_pos_vec4us(vec4us& x, vec4us* ret);
WP_API void builtin_pos_vec2ui(vec2ui& x, vec2ui* ret);
WP_API void builtin_pos_vec3ui(vec3ui& x, vec3ui* ret);
WP_API void builtin_pos_vec4ui(vec4ui& x, vec4ui* ret);
WP_API void builtin_pos_vec2ul(vec2ul& x, vec2ul* ret);
WP_API void builtin_pos_vec3ul(vec3ul& x, vec3ul* ret);
WP_API void builtin_pos_vec4ul(vec4ul& x, vec4ul* ret);
WP_API void builtin_pos_vec2ub(vec2ub& x, vec2ub* ret);
WP_API void builtin_pos_vec3ub(vec3ub& x, vec3ub* ret);
WP_API void builtin_pos_vec4ub(vec4ub& x, vec4ub* ret);
WP_API void builtin_pos_quath(quath& x, quath* ret);
WP_API void builtin_pos_quatf(quatf& x, quatf* ret);
WP_API void builtin_pos_quatd(quatd& x, quatd* ret);
WP_API void builtin_pos_mat22h(mat22h& x, mat22h* ret);
WP_API void builtin_pos_mat33h(mat33h& x, mat33h* ret);
WP_API void builtin_pos_mat44h(mat44h& x, mat44h* ret);
WP_API void builtin_pos_spatial_matrixh(spatial_matrixh& x, spatial_matrixh* ret);
WP_API void builtin_pos_mat22f(mat22f& x, mat22f* ret);
WP_API void builtin_pos_mat33f(mat33f& x, mat33f* ret);
WP_API void builtin_pos_mat44f(mat44f& x, mat44f* ret);
WP_API void builtin_pos_spatial_matrixf(spatial_matrixf& x, spatial_matrixf* ret);
WP_API void builtin_pos_mat22d(mat22d& x, mat22d* ret);
WP_API void builtin_pos_mat33d(mat33d& x, mat33d* ret);
WP_API void builtin_pos_mat44d(mat44d& x, mat44d* ret);
WP_API void builtin_pos_spatial_matrixd(spatial_matrixd& x, spatial_matrixd* ret);
WP_API void builtin_neg_float16(float16 x, float16* ret);
WP_API void builtin_neg_float32(float32 x, float32* ret);
WP_API void builtin_neg_float64(float64 x, float64* ret);
WP_API void builtin_neg_int16(int16 x, int16* ret);
WP_API void builtin_neg_int32(int32 x, int32* ret);
WP_API void builtin_neg_int64(int64 x, int64* ret);
WP_API void builtin_neg_int8(int8 x, int8* ret);
WP_API void builtin_neg_uint16(uint16 x, uint16* ret);
WP_API void builtin_neg_uint32(uint32 x, uint32* ret);
WP_API void builtin_neg_uint64(uint64 x, uint64* ret);
WP_API void builtin_neg_uint8(uint8 x, uint8* ret);
WP_API void builtin_neg_vec2h(vec2h& x, vec2h* ret);
WP_API void builtin_neg_vec3h(vec3h& x, vec3h* ret);
WP_API void builtin_neg_vec4h(vec4h& x, vec4h* ret);
WP_API void builtin_neg_spatial_vectorh(spatial_vectorh& x, spatial_vectorh* ret);
WP_API void builtin_neg_vec2f(vec2f& x, vec2f* ret);
WP_API void builtin_neg_vec3f(vec3f& x, vec3f* ret);
WP_API void builtin_neg_vec4f(vec4f& x, vec4f* ret);
WP_API void builtin_neg_spatial_vectorf(spatial_vectorf& x, spatial_vectorf* ret);
WP_API void builtin_neg_vec2d(vec2d& x, vec2d* ret);
WP_API void builtin_neg_vec3d(vec3d& x, vec3d* ret);
WP_API void builtin_neg_vec4d(vec4d& x, vec4d* ret);
WP_API void builtin_neg_spatial_vectord(spatial_vectord& x, spatial_vectord* ret);
WP_API void builtin_neg_vec2s(vec2s& x, vec2s* ret);
WP_API void builtin_neg_vec3s(vec3s& x, vec3s* ret);
WP_API void builtin_neg_vec4s(vec4s& x, vec4s* ret);
WP_API void builtin_neg_vec2i(vec2i& x, vec2i* ret);
WP_API void builtin_neg_vec3i(vec3i& x, vec3i* ret);
WP_API void builtin_neg_vec4i(vec4i& x, vec4i* ret);
WP_API void builtin_neg_vec2l(vec2l& x, vec2l* ret);
WP_API void builtin_neg_vec3l(vec3l& x, vec3l* ret);
WP_API void builtin_neg_vec4l(vec4l& x, vec4l* ret);
WP_API void builtin_neg_vec2b(vec2b& x, vec2b* ret);
WP_API void builtin_neg_vec3b(vec3b& x, vec3b* ret);
WP_API void builtin_neg_vec4b(vec4b& x, vec4b* ret);
WP_API void builtin_neg_vec2us(vec2us& x, vec2us* ret);
WP_API void builtin_neg_vec3us(vec3us& x, vec3us* ret);
WP_API void builtin_neg_vec4us(vec4us& x, vec4us* ret);
WP_API void builtin_neg_vec2ui(vec2ui& x, vec2ui* ret);
WP_API void builtin_neg_vec3ui(vec3ui& x, vec3ui* ret);
WP_API void builtin_neg_vec4ui(vec4ui& x, vec4ui* ret);
WP_API void builtin_neg_vec2ul(vec2ul& x, vec2ul* ret);
WP_API void builtin_neg_vec3ul(vec3ul& x, vec3ul* ret);
WP_API void builtin_neg_vec4ul(vec4ul& x, vec4ul* ret);
WP_API void builtin_neg_vec2ub(vec2ub& x, vec2ub* ret);
WP_API void builtin_neg_vec3ub(vec3ub& x, vec3ub* ret);
WP_API void builtin_neg_vec4ub(vec4ub& x, vec4ub* ret);
WP_API void builtin_neg_quath(quath& x, quath* ret);
WP_API void builtin_neg_quatf(quatf& x, quatf* ret);
WP_API void builtin_neg_quatd(quatd& x, quatd* ret);
WP_API void builtin_neg_mat22h(mat22h& x, mat22h* ret);
WP_API void builtin_neg_mat33h(mat33h& x, mat33h* ret);
WP_API void builtin_neg_mat44h(mat44h& x, mat44h* ret);
WP_API void builtin_neg_spatial_matrixh(spatial_matrixh& x, spatial_matrixh* ret);
WP_API void builtin_neg_mat22f(mat22f& x, mat22f* ret);
WP_API void builtin_neg_mat33f(mat33f& x, mat33f* ret);
WP_API void builtin_neg_mat44f(mat44f& x, mat44f* ret);
WP_API void builtin_neg_spatial_matrixf(spatial_matrixf& x, spatial_matrixf* ret);
WP_API void builtin_neg_mat22d(mat22d& x, mat22d* ret);
WP_API void builtin_neg_mat33d(mat33d& x, mat33d* ret);
WP_API void builtin_neg_mat44d(mat44d& x, mat44d* ret);
WP_API void builtin_neg_spatial_matrixd(spatial_matrixd& x, spatial_matrixd* ret);
WP_API void builtin_unot_bool(bool a, bool* ret);
WP_API void builtin_unot_int8(int8 a, bool* ret);
WP_API void builtin_unot_uint8(uint8 a, bool* ret);
WP_API void builtin_unot_int16(int16 a, bool* ret);
WP_API void builtin_unot_uint16(uint16 a, bool* ret);
WP_API void builtin_unot_int32(int32 a, bool* ret);
WP_API void builtin_unot_uint32(uint32 a, bool* ret);
WP_API void builtin_unot_int64(int64 a, bool* ret);
WP_API void builtin_unot_uint64(uint64 a, bool* ret);

}  // extern "C"

WP_API void builtin_min_float16_float16(float16 a, float16 b, float16* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_float32_float32(float32 a, float32 b, float32* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_float64_float64(float64 a, float64 b, float64* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret) { *ret = wp::min(a, b); }
WP_API void builtin_min_vec2h(vec2h& a, float16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3h(vec3h& a, float16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4h(vec4h& a, float16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_spatial_vectorh(spatial_vectorh& a, float16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2f(vec2f& a, float32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3f(vec3f& a, float32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4f(vec4f& a, float32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_spatial_vectorf(spatial_vectorf& a, float32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2d(vec2d& a, float64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3d(vec3d& a, float64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4d(vec4d& a, float64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_spatial_vectord(spatial_vectord& a, float64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2s(vec2s& a, int16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3s(vec3s& a, int16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4s(vec4s& a, int16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2i(vec2i& a, int32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3i(vec3i& a, int32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4i(vec4i& a, int32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2l(vec2l& a, int64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3l(vec3l& a, int64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4l(vec4l& a, int64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2b(vec2b& a, int8* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3b(vec3b& a, int8* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4b(vec4b& a, int8* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2us(vec2us& a, uint16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3us(vec3us& a, uint16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4us(vec4us& a, uint16* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2ui(vec2ui& a, uint32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3ui(vec3ui& a, uint32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4ui(vec4ui& a, uint32* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2ul(vec2ul& a, uint64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3ul(vec3ul& a, uint64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4ul(vec4ul& a, uint64* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec2ub(vec2ub& a, uint8* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec3ub(vec3ub& a, uint8* ret) { *ret = wp::min(a); }
WP_API void builtin_min_vec4ub(vec4ub& a, uint8* ret) { *ret = wp::min(a); }
WP_API void builtin_max_float16_float16(float16 a, float16 b, float16* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_float32_float32(float32 a, float32 b, float32* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_float64_float64(float64 a, float64 b, float64* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret) { *ret = wp::max(a, b); }
WP_API void builtin_max_vec2h(vec2h& a, float16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3h(vec3h& a, float16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4h(vec4h& a, float16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_spatial_vectorh(spatial_vectorh& a, float16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2f(vec2f& a, float32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3f(vec3f& a, float32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4f(vec4f& a, float32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_spatial_vectorf(spatial_vectorf& a, float32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2d(vec2d& a, float64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3d(vec3d& a, float64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4d(vec4d& a, float64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_spatial_vectord(spatial_vectord& a, float64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2s(vec2s& a, int16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3s(vec3s& a, int16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4s(vec4s& a, int16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2i(vec2i& a, int32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3i(vec3i& a, int32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4i(vec4i& a, int32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2l(vec2l& a, int64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3l(vec3l& a, int64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4l(vec4l& a, int64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2b(vec2b& a, int8* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3b(vec3b& a, int8* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4b(vec4b& a, int8* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2us(vec2us& a, uint16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3us(vec3us& a, uint16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4us(vec4us& a, uint16* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2ui(vec2ui& a, uint32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3ui(vec3ui& a, uint32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4ui(vec4ui& a, uint32* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2ul(vec2ul& a, uint64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3ul(vec3ul& a, uint64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4ul(vec4ul& a, uint64* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec2ub(vec2ub& a, uint8* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec3ub(vec3ub& a, uint8* ret) { *ret = wp::max(a); }
WP_API void builtin_max_vec4ub(vec4ub& a, uint8* ret) { *ret = wp::max(a); }
WP_API void builtin_clamp_float16_float16_float16(float16 x, float16 low, float16 high, float16* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_float32_float32_float32(float32 x, float32 low, float32 high, float32* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_float64_float64_float64(float64 x, float64 low, float64 high, float64* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_int16_int16_int16(int16 x, int16 low, int16 high, int16* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_int32_int32_int32(int32 x, int32 low, int32 high, int32* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_int64_int64_int64(int64 x, int64 low, int64 high, int64* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_int8_int8_int8(int8 x, int8 low, int8 high, int8* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_uint16_uint16_uint16(uint16 x, uint16 low, uint16 high, uint16* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_uint32_uint32_uint32(uint32 x, uint32 low, uint32 high, uint32* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_uint64_uint64_uint64(uint64 x, uint64 low, uint64 high, uint64* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_clamp_uint8_uint8_uint8(uint8 x, uint8 low, uint8 high, uint8* ret) { *ret = wp::clamp(x, low, high); }
WP_API void builtin_abs_float16(float16 x, float16* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_float32(float32 x, float32* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_float64(float64 x, float64* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_int16(int16 x, int16* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_int32(int32 x, int32* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_int64(int64 x, int64* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_int8(int8 x, int8* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_uint16(uint16 x, uint16* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_uint32(uint32 x, uint32* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_uint64(uint64 x, uint64* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_uint8(uint8 x, uint8* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2h(vec2h& x, vec2h* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3h(vec3h& x, vec3h* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4h(vec4h& x, vec4h* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_spatial_vectorh(spatial_vectorh& x, spatial_vectorh* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2f(vec2f& x, vec2f* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3f(vec3f& x, vec3f* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4f(vec4f& x, vec4f* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_spatial_vectorf(spatial_vectorf& x, spatial_vectorf* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2d(vec2d& x, vec2d* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3d(vec3d& x, vec3d* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4d(vec4d& x, vec4d* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_spatial_vectord(spatial_vectord& x, spatial_vectord* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2s(vec2s& x, vec2s* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3s(vec3s& x, vec3s* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4s(vec4s& x, vec4s* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2i(vec2i& x, vec2i* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3i(vec3i& x, vec3i* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4i(vec4i& x, vec4i* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2l(vec2l& x, vec2l* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3l(vec3l& x, vec3l* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4l(vec4l& x, vec4l* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2b(vec2b& x, vec2b* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3b(vec3b& x, vec3b* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4b(vec4b& x, vec4b* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2us(vec2us& x, vec2us* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3us(vec3us& x, vec3us* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4us(vec4us& x, vec4us* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2ui(vec2ui& x, vec2ui* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3ui(vec3ui& x, vec3ui* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4ui(vec4ui& x, vec4ui* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2ul(vec2ul& x, vec2ul* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3ul(vec3ul& x, vec3ul* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4ul(vec4ul& x, vec4ul* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec2ub(vec2ub& x, vec2ub* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec3ub(vec3ub& x, vec3ub* ret) { *ret = wp::abs(x); }
WP_API void builtin_abs_vec4ub(vec4ub& x, vec4ub* ret) { *ret = wp::abs(x); }
WP_API void builtin_sign_float16(float16 x, float16* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_float32(float32 x, float32* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_float64(float64 x, float64* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_int16(int16 x, int16* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_int32(int32 x, int32* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_int64(int64 x, int64* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_int8(int8 x, int8* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_uint16(uint16 x, uint16* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_uint32(uint32 x, uint32* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_uint64(uint64 x, uint64* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_uint8(uint8 x, uint8* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2h(vec2h& x, vec2h* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3h(vec3h& x, vec3h* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4h(vec4h& x, vec4h* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_spatial_vectorh(spatial_vectorh& x, spatial_vectorh* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2f(vec2f& x, vec2f* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3f(vec3f& x, vec3f* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4f(vec4f& x, vec4f* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_spatial_vectorf(spatial_vectorf& x, spatial_vectorf* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2d(vec2d& x, vec2d* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3d(vec3d& x, vec3d* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4d(vec4d& x, vec4d* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_spatial_vectord(spatial_vectord& x, spatial_vectord* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2s(vec2s& x, vec2s* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3s(vec3s& x, vec3s* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4s(vec4s& x, vec4s* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2i(vec2i& x, vec2i* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3i(vec3i& x, vec3i* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4i(vec4i& x, vec4i* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2l(vec2l& x, vec2l* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3l(vec3l& x, vec3l* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4l(vec4l& x, vec4l* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2b(vec2b& x, vec2b* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3b(vec3b& x, vec3b* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4b(vec4b& x, vec4b* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2us(vec2us& x, vec2us* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3us(vec3us& x, vec3us* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4us(vec4us& x, vec4us* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2ui(vec2ui& x, vec2ui* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3ui(vec3ui& x, vec3ui* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4ui(vec4ui& x, vec4ui* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2ul(vec2ul& x, vec2ul* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3ul(vec3ul& x, vec3ul* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4ul(vec4ul& x, vec4ul* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec2ub(vec2ub& x, vec2ub* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec3ub(vec3ub& x, vec3ub* ret) { *ret = wp::sign(x); }
WP_API void builtin_sign_vec4ub(vec4ub& x, vec4ub* ret) { *ret = wp::sign(x); }
WP_API void builtin_step_float16(float16 x, float16* ret) { *ret = wp::step(x); }
WP_API void builtin_step_float32(float32 x, float32* ret) { *ret = wp::step(x); }
WP_API void builtin_step_float64(float64 x, float64* ret) { *ret = wp::step(x); }
WP_API void builtin_step_int16(int16 x, int16* ret) { *ret = wp::step(x); }
WP_API void builtin_step_int32(int32 x, int32* ret) { *ret = wp::step(x); }
WP_API void builtin_step_int64(int64 x, int64* ret) { *ret = wp::step(x); }
WP_API void builtin_step_int8(int8 x, int8* ret) { *ret = wp::step(x); }
WP_API void builtin_step_uint16(uint16 x, uint16* ret) { *ret = wp::step(x); }
WP_API void builtin_step_uint32(uint32 x, uint32* ret) { *ret = wp::step(x); }
WP_API void builtin_step_uint64(uint64 x, uint64* ret) { *ret = wp::step(x); }
WP_API void builtin_step_uint8(uint8 x, uint8* ret) { *ret = wp::step(x); }
WP_API void builtin_nonzero_float16(float16 x, float16* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_float32(float32 x, float32* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_float64(float64 x, float64* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_int16(int16 x, int16* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_int32(int32 x, int32* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_int64(int64 x, int64* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_int8(int8 x, int8* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_uint16(uint16 x, uint16* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_uint32(uint32 x, uint32* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_uint64(uint64 x, uint64* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_nonzero_uint8(uint8 x, uint8* ret) { *ret = wp::nonzero(x); }
WP_API void builtin_sin_float16(float16 x, float16* ret) { *ret = wp::sin(x); }
WP_API void builtin_sin_float32(float32 x, float32* ret) { *ret = wp::sin(x); }
WP_API void builtin_sin_float64(float64 x, float64* ret) { *ret = wp::sin(x); }
WP_API void builtin_cos_float16(float16 x, float16* ret) { *ret = wp::cos(x); }
WP_API void builtin_cos_float32(float32 x, float32* ret) { *ret = wp::cos(x); }
WP_API void builtin_cos_float64(float64 x, float64* ret) { *ret = wp::cos(x); }
WP_API void builtin_acos_float16(float16 x, float16* ret) { *ret = wp::acos(x); }
WP_API void builtin_acos_float32(float32 x, float32* ret) { *ret = wp::acos(x); }
WP_API void builtin_acos_float64(float64 x, float64* ret) { *ret = wp::acos(x); }
WP_API void builtin_asin_float16(float16 x, float16* ret) { *ret = wp::asin(x); }
WP_API void builtin_asin_float32(float32 x, float32* ret) { *ret = wp::asin(x); }
WP_API void builtin_asin_float64(float64 x, float64* ret) { *ret = wp::asin(x); }
WP_API void builtin_sqrt_float16(float16 x, float16* ret) { *ret = wp::sqrt(x); }
WP_API void builtin_sqrt_float32(float32 x, float32* ret) { *ret = wp::sqrt(x); }
WP_API void builtin_sqrt_float64(float64 x, float64* ret) { *ret = wp::sqrt(x); }
WP_API void builtin_cbrt_float16(float16 x, float16* ret) { *ret = wp::cbrt(x); }
WP_API void builtin_cbrt_float32(float32 x, float32* ret) { *ret = wp::cbrt(x); }
WP_API void builtin_cbrt_float64(float64 x, float64* ret) { *ret = wp::cbrt(x); }
WP_API void builtin_tan_float16(float16 x, float16* ret) { *ret = wp::tan(x); }
WP_API void builtin_tan_float32(float32 x, float32* ret) { *ret = wp::tan(x); }
WP_API void builtin_tan_float64(float64 x, float64* ret) { *ret = wp::tan(x); }
WP_API void builtin_atan_float16(float16 x, float16* ret) { *ret = wp::atan(x); }
WP_API void builtin_atan_float32(float32 x, float32* ret) { *ret = wp::atan(x); }
WP_API void builtin_atan_float64(float64 x, float64* ret) { *ret = wp::atan(x); }
WP_API void builtin_atan2_float16_float16(float16 y, float16 x, float16* ret) { *ret = wp::atan2(y, x); }
WP_API void builtin_atan2_float32_float32(float32 y, float32 x, float32* ret) { *ret = wp::atan2(y, x); }
WP_API void builtin_atan2_float64_float64(float64 y, float64 x, float64* ret) { *ret = wp::atan2(y, x); }
WP_API void builtin_sinh_float16(float16 x, float16* ret) { *ret = wp::sinh(x); }
WP_API void builtin_sinh_float32(float32 x, float32* ret) { *ret = wp::sinh(x); }
WP_API void builtin_sinh_float64(float64 x, float64* ret) { *ret = wp::sinh(x); }
WP_API void builtin_cosh_float16(float16 x, float16* ret) { *ret = wp::cosh(x); }
WP_API void builtin_cosh_float32(float32 x, float32* ret) { *ret = wp::cosh(x); }
WP_API void builtin_cosh_float64(float64 x, float64* ret) { *ret = wp::cosh(x); }
WP_API void builtin_tanh_float16(float16 x, float16* ret) { *ret = wp::tanh(x); }
WP_API void builtin_tanh_float32(float32 x, float32* ret) { *ret = wp::tanh(x); }
WP_API void builtin_tanh_float64(float64 x, float64* ret) { *ret = wp::tanh(x); }
WP_API void builtin_degrees_float16(float16 x, float16* ret) { *ret = wp::degrees(x); }
WP_API void builtin_degrees_float32(float32 x, float32* ret) { *ret = wp::degrees(x); }
WP_API void builtin_degrees_float64(float64 x, float64* ret) { *ret = wp::degrees(x); }
WP_API void builtin_radians_float16(float16 x, float16* ret) { *ret = wp::radians(x); }
WP_API void builtin_radians_float32(float32 x, float32* ret) { *ret = wp::radians(x); }
WP_API void builtin_radians_float64(float64 x, float64* ret) { *ret = wp::radians(x); }
WP_API void builtin_log_float16(float16 x, float16* ret) { *ret = wp::log(x); }
WP_API void builtin_log_float32(float32 x, float32* ret) { *ret = wp::log(x); }
WP_API void builtin_log_float64(float64 x, float64* ret) { *ret = wp::log(x); }
WP_API void builtin_log2_float16(float16 x, float16* ret) { *ret = wp::log2(x); }
WP_API void builtin_log2_float32(float32 x, float32* ret) { *ret = wp::log2(x); }
WP_API void builtin_log2_float64(float64 x, float64* ret) { *ret = wp::log2(x); }
WP_API void builtin_log10_float16(float16 x, float16* ret) { *ret = wp::log10(x); }
WP_API void builtin_log10_float32(float32 x, float32* ret) { *ret = wp::log10(x); }
WP_API void builtin_log10_float64(float64 x, float64* ret) { *ret = wp::log10(x); }
WP_API void builtin_exp_float16(float16 x, float16* ret) { *ret = wp::exp(x); }
WP_API void builtin_exp_float32(float32 x, float32* ret) { *ret = wp::exp(x); }
WP_API void builtin_exp_float64(float64 x, float64* ret) { *ret = wp::exp(x); }
WP_API void builtin_pow_float16_float16(float16 x, float16 y, float16* ret) { *ret = wp::pow(x, y); }
WP_API void builtin_pow_float32_float32(float32 x, float32 y, float32* ret) { *ret = wp::pow(x, y); }
WP_API void builtin_pow_float64_float64(float64 x, float64 y, float64* ret) { *ret = wp::pow(x, y); }
WP_API void builtin_round_float16(float16 x, float16* ret) { *ret = wp::round(x); }
WP_API void builtin_round_float32(float32 x, float32* ret) { *ret = wp::round(x); }
WP_API void builtin_round_float64(float64 x, float64* ret) { *ret = wp::round(x); }
WP_API void builtin_rint_float16(float16 x, float16* ret) { *ret = wp::rint(x); }
WP_API void builtin_rint_float32(float32 x, float32* ret) { *ret = wp::rint(x); }
WP_API void builtin_rint_float64(float64 x, float64* ret) { *ret = wp::rint(x); }
WP_API void builtin_trunc_float16(float16 x, float16* ret) { *ret = wp::trunc(x); }
WP_API void builtin_trunc_float32(float32 x, float32* ret) { *ret = wp::trunc(x); }
WP_API void builtin_trunc_float64(float64 x, float64* ret) { *ret = wp::trunc(x); }
WP_API void builtin_floor_float16(float16 x, float16* ret) { *ret = wp::floor(x); }
WP_API void builtin_floor_float32(float32 x, float32* ret) { *ret = wp::floor(x); }
WP_API void builtin_floor_float64(float64 x, float64* ret) { *ret = wp::floor(x); }
WP_API void builtin_ceil_float16(float16 x, float16* ret) { *ret = wp::ceil(x); }
WP_API void builtin_ceil_float32(float32 x, float32* ret) { *ret = wp::ceil(x); }
WP_API void builtin_ceil_float64(float64 x, float64* ret) { *ret = wp::ceil(x); }
WP_API void builtin_frac_float16(float16 x, float16* ret) { *ret = wp::frac(x); }
WP_API void builtin_frac_float32(float32 x, float32* ret) { *ret = wp::frac(x); }
WP_API void builtin_frac_float64(float64 x, float64* ret) { *ret = wp::frac(x); }
WP_API void builtin_isfinite_float16(float16 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_float32(float32 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_float64(float64 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_int16(int16 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_int32(int32 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_int64(int64 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_int8(int8 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_uint16(uint16 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_uint32(uint32 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_uint64(uint64 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_uint8(uint8 a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2h(vec2h& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3h(vec3h& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4h(vec4h& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_spatial_vectorh(spatial_vectorh& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2f(vec2f& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3f(vec3f& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4f(vec4f& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_spatial_vectorf(spatial_vectorf& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2d(vec2d& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3d(vec3d& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4d(vec4d& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_spatial_vectord(spatial_vectord& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2s(vec2s& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3s(vec3s& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4s(vec4s& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2i(vec2i& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3i(vec3i& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4i(vec4i& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2l(vec2l& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3l(vec3l& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4l(vec4l& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2b(vec2b& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3b(vec3b& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4b(vec4b& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2us(vec2us& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3us(vec3us& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4us(vec4us& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2ui(vec2ui& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3ui(vec3ui& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4ui(vec4ui& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2ul(vec2ul& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3ul(vec3ul& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4ul(vec4ul& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec2ub(vec2ub& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec3ub(vec3ub& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_vec4ub(vec4ub& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_quath(quath& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_quatf(quatf& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_quatd(quatd& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat22h(mat22h& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat33h(mat33h& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat44h(mat44h& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_spatial_matrixh(spatial_matrixh& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat22f(mat22f& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat33f(mat33f& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat44f(mat44f& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_spatial_matrixf(spatial_matrixf& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat22d(mat22d& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat33d(mat33d& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_mat44d(mat44d& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isfinite_spatial_matrixd(spatial_matrixd& a, bool* ret) { *ret = wp::isfinite(a); }
WP_API void builtin_isnan_float16(float16 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_float32(float32 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_float64(float64 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_int16(int16 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_int32(int32 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_int64(int64 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_int8(int8 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_uint16(uint16 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_uint32(uint32 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_uint64(uint64 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_uint8(uint8 a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2h(vec2h& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3h(vec3h& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4h(vec4h& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_spatial_vectorh(spatial_vectorh& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2f(vec2f& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3f(vec3f& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4f(vec4f& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_spatial_vectorf(spatial_vectorf& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2d(vec2d& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3d(vec3d& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4d(vec4d& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_spatial_vectord(spatial_vectord& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2s(vec2s& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3s(vec3s& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4s(vec4s& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2i(vec2i& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3i(vec3i& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4i(vec4i& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2l(vec2l& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3l(vec3l& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4l(vec4l& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2b(vec2b& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3b(vec3b& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4b(vec4b& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2us(vec2us& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3us(vec3us& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4us(vec4us& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2ui(vec2ui& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3ui(vec3ui& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4ui(vec4ui& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2ul(vec2ul& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3ul(vec3ul& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4ul(vec4ul& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec2ub(vec2ub& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec3ub(vec3ub& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_vec4ub(vec4ub& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_quath(quath& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_quatf(quatf& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_quatd(quatd& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat22h(mat22h& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat33h(mat33h& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat44h(mat44h& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_spatial_matrixh(spatial_matrixh& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat22f(mat22f& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat33f(mat33f& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat44f(mat44f& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_spatial_matrixf(spatial_matrixf& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat22d(mat22d& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat33d(mat33d& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_mat44d(mat44d& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isnan_spatial_matrixd(spatial_matrixd& a, bool* ret) { *ret = wp::isnan(a); }
WP_API void builtin_isinf_float16(float16 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_float32(float32 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_float64(float64 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_int16(int16 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_int32(int32 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_int64(int64 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_int8(int8 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_uint16(uint16 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_uint32(uint32 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_uint64(uint64 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_uint8(uint8 a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2h(vec2h& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3h(vec3h& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4h(vec4h& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_spatial_vectorh(spatial_vectorh& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2f(vec2f& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3f(vec3f& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4f(vec4f& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_spatial_vectorf(spatial_vectorf& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2d(vec2d& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3d(vec3d& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4d(vec4d& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_spatial_vectord(spatial_vectord& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2s(vec2s& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3s(vec3s& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4s(vec4s& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2i(vec2i& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3i(vec3i& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4i(vec4i& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2l(vec2l& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3l(vec3l& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4l(vec4l& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2b(vec2b& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3b(vec3b& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4b(vec4b& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2us(vec2us& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3us(vec3us& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4us(vec4us& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2ui(vec2ui& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3ui(vec3ui& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4ui(vec4ui& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2ul(vec2ul& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3ul(vec3ul& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4ul(vec4ul& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec2ub(vec2ub& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec3ub(vec3ub& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_vec4ub(vec4ub& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_quath(quath& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_quatf(quatf& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_quatd(quatd& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat22h(mat22h& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat33h(mat33h& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat44h(mat44h& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_spatial_matrixh(spatial_matrixh& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat22f(mat22f& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat33f(mat33f& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat44f(mat44f& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_spatial_matrixf(spatial_matrixf& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat22d(mat22d& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat33d(mat33d& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_mat44d(mat44d& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_isinf_spatial_matrixd(spatial_matrixd& a, bool* ret) { *ret = wp::isinf(a); }
WP_API void builtin_dot_vec2h_vec2h(vec2h& a, vec2h& b, float16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3h_vec3h(vec3h& a, vec3h& b, float16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4h_vec4h(vec4h& a, vec4h& b, float16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, float16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2f_vec2f(vec2f& a, vec2f& b, float32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3f_vec3f(vec3f& a, vec3f& b, float32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4f_vec4f(vec4f& a, vec4f& b, float32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, float32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2d_vec2d(vec2d& a, vec2d& b, float64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3d_vec3d(vec3d& a, vec3d& b, float64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4d_vec4d(vec4d& a, vec4d& b, float64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, float64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2s_vec2s(vec2s& a, vec2s& b, int16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3s_vec3s(vec3s& a, vec3s& b, int16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4s_vec4s(vec4s& a, vec4s& b, int16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2i_vec2i(vec2i& a, vec2i& b, int32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3i_vec3i(vec3i& a, vec3i& b, int32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4i_vec4i(vec4i& a, vec4i& b, int32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2l_vec2l(vec2l& a, vec2l& b, int64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3l_vec3l(vec3l& a, vec3l& b, int64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4l_vec4l(vec4l& a, vec4l& b, int64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2b_vec2b(vec2b& a, vec2b& b, int8* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3b_vec3b(vec3b& a, vec3b& b, int8* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4b_vec4b(vec4b& a, vec4b& b, int8* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2us_vec2us(vec2us& a, vec2us& b, uint16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3us_vec3us(vec3us& a, vec3us& b, uint16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4us_vec4us(vec4us& a, vec4us& b, uint16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2ui_vec2ui(vec2ui& a, vec2ui& b, uint32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3ui_vec3ui(vec3ui& a, vec3ui& b, uint32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4ui_vec4ui(vec4ui& a, vec4ui& b, uint32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2ul_vec2ul(vec2ul& a, vec2ul& b, uint64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3ul_vec3ul(vec3ul& a, vec3ul& b, uint64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4ul_vec4ul(vec4ul& a, vec4ul& b, uint64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec2ub_vec2ub(vec2ub& a, vec2ub& b, uint8* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec3ub_vec3ub(vec3ub& a, vec3ub& b, uint8* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_vec4ub_vec4ub(vec4ub& a, vec4ub& b, uint8* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_quath_quath(quath& a, quath& b, float16* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_quatf_quatf(quatf& a, quatf& b, float32* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_dot_quatd_quatd(quatd& a, quatd& b, float64* ret) { *ret = wp::dot(a, b); }
WP_API void builtin_ddot_mat22h_mat22h(mat22h& a, mat22h& b, float16* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_mat33h_mat33h(mat33h& a, mat33h& b, float16* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_mat44h_mat44h(mat44h& a, mat44h& b, float16* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, float16* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_mat22f_mat22f(mat22f& a, mat22f& b, float32* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_mat33f_mat33f(mat33f& a, mat33f& b, float32* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_mat44f_mat44f(mat44f& a, mat44f& b, float32* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, float32* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_mat22d_mat22d(mat22d& a, mat22d& b, float64* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_mat33d_mat33d(mat33d& a, mat33d& b, float64* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_mat44d_mat44d(mat44d& a, mat44d& b, float64* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_ddot_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, float64* ret) { *ret = wp::ddot(a, b); }
WP_API void builtin_argmin_vec2h(vec2h& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3h(vec3h& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4h(vec4h& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_spatial_vectorh(spatial_vectorh& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2f(vec2f& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3f(vec3f& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4f(vec4f& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_spatial_vectorf(spatial_vectorf& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2d(vec2d& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3d(vec3d& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4d(vec4d& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_spatial_vectord(spatial_vectord& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2s(vec2s& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3s(vec3s& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4s(vec4s& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2i(vec2i& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3i(vec3i& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4i(vec4i& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2l(vec2l& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3l(vec3l& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4l(vec4l& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2b(vec2b& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3b(vec3b& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4b(vec4b& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2us(vec2us& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3us(vec3us& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4us(vec4us& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2ui(vec2ui& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3ui(vec3ui& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4ui(vec4ui& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2ul(vec2ul& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3ul(vec3ul& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4ul(vec4ul& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec2ub(vec2ub& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec3ub(vec3ub& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmin_vec4ub(vec4ub& a, uint32* ret) { *ret = wp::argmin(a); }
WP_API void builtin_argmax_vec2h(vec2h& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3h(vec3h& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4h(vec4h& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_spatial_vectorh(spatial_vectorh& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2f(vec2f& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3f(vec3f& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4f(vec4f& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_spatial_vectorf(spatial_vectorf& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2d(vec2d& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3d(vec3d& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4d(vec4d& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_spatial_vectord(spatial_vectord& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2s(vec2s& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3s(vec3s& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4s(vec4s& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2i(vec2i& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3i(vec3i& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4i(vec4i& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2l(vec2l& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3l(vec3l& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4l(vec4l& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2b(vec2b& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3b(vec3b& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4b(vec4b& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2us(vec2us& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3us(vec3us& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4us(vec4us& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2ui(vec2ui& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3ui(vec3ui& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4ui(vec4ui& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2ul(vec2ul& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3ul(vec3ul& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4ul(vec4ul& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec2ub(vec2ub& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec3ub(vec3ub& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_argmax_vec4ub(vec4ub& a, uint32* ret) { *ret = wp::argmax(a); }
WP_API void builtin_outer_vec2h_vec2h(vec2h& a, vec2h& b, mat22h* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_vec3h_vec3h(vec3h& a, vec3h& b, mat33h* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_vec4h_vec4h(vec4h& a, vec4h& b, mat44h* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_matrixh* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_vec2f_vec2f(vec2f& a, vec2f& b, mat22f* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_vec3f_vec3f(vec3f& a, vec3f& b, mat33f* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_vec4f_vec4f(vec4f& a, vec4f& b, mat44f* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_matrixf* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_vec2d_vec2d(vec2d& a, vec2d& b, mat22d* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_vec3d_vec3d(vec3d& a, vec3d& b, mat33d* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_vec4d_vec4d(vec4d& a, vec4d& b, mat44d* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_outer_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_matrixd* ret) { *ret = wp::outer(a, b); }
WP_API void builtin_cross_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_cross_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret) { *ret = wp::cross(a, b); }
WP_API void builtin_skew_vec3h(vec3h& vec, mat33h* ret) { *ret = wp::skew(vec); }
WP_API void builtin_skew_vec3f(vec3f& vec, mat33f* ret) { *ret = wp::skew(vec); }
WP_API void builtin_skew_vec3d(vec3d& vec, mat33d* ret) { *ret = wp::skew(vec); }
WP_API void builtin_length_vec2h(vec2h& a, float16* ret) { *ret = wp::length(a); }
WP_API void builtin_length_vec3h(vec3h& a, float16* ret) { *ret = wp::length(a); }
WP_API void builtin_length_vec4h(vec4h& a, float16* ret) { *ret = wp::length(a); }
WP_API void builtin_length_spatial_vectorh(spatial_vectorh& a, float16* ret) { *ret = wp::length(a); }
WP_API void builtin_length_vec2f(vec2f& a, float32* ret) { *ret = wp::length(a); }
WP_API void builtin_length_vec3f(vec3f& a, float32* ret) { *ret = wp::length(a); }
WP_API void builtin_length_vec4f(vec4f& a, float32* ret) { *ret = wp::length(a); }
WP_API void builtin_length_spatial_vectorf(spatial_vectorf& a, float32* ret) { *ret = wp::length(a); }
WP_API void builtin_length_vec2d(vec2d& a, float64* ret) { *ret = wp::length(a); }
WP_API void builtin_length_vec3d(vec3d& a, float64* ret) { *ret = wp::length(a); }
WP_API void builtin_length_vec4d(vec4d& a, float64* ret) { *ret = wp::length(a); }
WP_API void builtin_length_spatial_vectord(spatial_vectord& a, float64* ret) { *ret = wp::length(a); }
WP_API void builtin_length_quath(quath& a, float16* ret) { *ret = wp::length(a); }
WP_API void builtin_length_quatf(quatf& a, float32* ret) { *ret = wp::length(a); }
WP_API void builtin_length_quatd(quatd& a, float64* ret) { *ret = wp::length(a); }
WP_API void builtin_length_sq_vec2h(vec2h& a, float16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3h(vec3h& a, float16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4h(vec4h& a, float16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_spatial_vectorh(spatial_vectorh& a, float16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2f(vec2f& a, float32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3f(vec3f& a, float32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4f(vec4f& a, float32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_spatial_vectorf(spatial_vectorf& a, float32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2d(vec2d& a, float64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3d(vec3d& a, float64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4d(vec4d& a, float64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_spatial_vectord(spatial_vectord& a, float64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2s(vec2s& a, int16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3s(vec3s& a, int16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4s(vec4s& a, int16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2i(vec2i& a, int32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3i(vec3i& a, int32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4i(vec4i& a, int32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2l(vec2l& a, int64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3l(vec3l& a, int64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4l(vec4l& a, int64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2b(vec2b& a, int8* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3b(vec3b& a, int8* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4b(vec4b& a, int8* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2us(vec2us& a, uint16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3us(vec3us& a, uint16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4us(vec4us& a, uint16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2ui(vec2ui& a, uint32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3ui(vec3ui& a, uint32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4ui(vec4ui& a, uint32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2ul(vec2ul& a, uint64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3ul(vec3ul& a, uint64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4ul(vec4ul& a, uint64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec2ub(vec2ub& a, uint8* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec3ub(vec3ub& a, uint8* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_vec4ub(vec4ub& a, uint8* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_quath(quath& a, float16* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_quatf(quatf& a, float32* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_length_sq_quatd(quatd& a, float64* ret) { *ret = wp::length_sq(a); }
WP_API void builtin_normalize_vec2h(vec2h& a, vec2h* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_vec3h(vec3h& a, vec3h* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_vec4h(vec4h& a, vec4h* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_spatial_vectorh(spatial_vectorh& a, spatial_vectorh* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_vec2f(vec2f& a, vec2f* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_vec3f(vec3f& a, vec3f* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_vec4f(vec4f& a, vec4f* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_spatial_vectorf(spatial_vectorf& a, spatial_vectorf* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_vec2d(vec2d& a, vec2d* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_vec3d(vec3d& a, vec3d* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_vec4d(vec4d& a, vec4d* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_spatial_vectord(spatial_vectord& a, spatial_vectord* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_quath(quath& a, quath* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_quatf(quatf& a, quatf* ret) { *ret = wp::normalize(a); }
WP_API void builtin_normalize_quatd(quatd& a, quatd* ret) { *ret = wp::normalize(a); }
WP_API void builtin_transpose_mat22h(mat22h& a, mat22h* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_mat33h(mat33h& a, mat33h* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_mat44h(mat44h& a, mat44h* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_spatial_matrixh(spatial_matrixh& a, spatial_matrixh* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_mat22f(mat22f& a, mat22f* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_mat33f(mat33f& a, mat33f* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_mat44f(mat44f& a, mat44f* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_spatial_matrixf(spatial_matrixf& a, spatial_matrixf* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_mat22d(mat22d& a, mat22d* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_mat33d(mat33d& a, mat33d* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_mat44d(mat44d& a, mat44d* ret) { *ret = wp::transpose(a); }
WP_API void builtin_transpose_spatial_matrixd(spatial_matrixd& a, spatial_matrixd* ret) { *ret = wp::transpose(a); }
WP_API void builtin_inverse_mat22h(mat22h& a, mat22h* ret) { *ret = wp::inverse(a); }
WP_API void builtin_inverse_mat22f(mat22f& a, mat22f* ret) { *ret = wp::inverse(a); }
WP_API void builtin_inverse_mat22d(mat22d& a, mat22d* ret) { *ret = wp::inverse(a); }
WP_API void builtin_inverse_mat33h(mat33h& a, mat33h* ret) { *ret = wp::inverse(a); }
WP_API void builtin_inverse_mat33f(mat33f& a, mat33f* ret) { *ret = wp::inverse(a); }
WP_API void builtin_inverse_mat33d(mat33d& a, mat33d* ret) { *ret = wp::inverse(a); }
WP_API void builtin_inverse_mat44h(mat44h& a, mat44h* ret) { *ret = wp::inverse(a); }
WP_API void builtin_inverse_mat44f(mat44f& a, mat44f* ret) { *ret = wp::inverse(a); }
WP_API void builtin_inverse_mat44d(mat44d& a, mat44d* ret) { *ret = wp::inverse(a); }
WP_API void builtin_determinant_mat22h(mat22h& a, float16* ret) { *ret = wp::determinant(a); }
WP_API void builtin_determinant_mat22f(mat22f& a, float32* ret) { *ret = wp::determinant(a); }
WP_API void builtin_determinant_mat22d(mat22d& a, float64* ret) { *ret = wp::determinant(a); }
WP_API void builtin_determinant_mat33h(mat33h& a, float16* ret) { *ret = wp::determinant(a); }
WP_API void builtin_determinant_mat33f(mat33f& a, float32* ret) { *ret = wp::determinant(a); }
WP_API void builtin_determinant_mat33d(mat33d& a, float64* ret) { *ret = wp::determinant(a); }
WP_API void builtin_determinant_mat44h(mat44h& a, float16* ret) { *ret = wp::determinant(a); }
WP_API void builtin_determinant_mat44f(mat44f& a, float32* ret) { *ret = wp::determinant(a); }
WP_API void builtin_determinant_mat44d(mat44d& a, float64* ret) { *ret = wp::determinant(a); }
WP_API void builtin_trace_mat22h(mat22h& a, float16* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_mat33h(mat33h& a, float16* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_mat44h(mat44h& a, float16* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_spatial_matrixh(spatial_matrixh& a, float16* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_mat22f(mat22f& a, float32* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_mat33f(mat33f& a, float32* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_mat44f(mat44f& a, float32* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_spatial_matrixf(spatial_matrixf& a, float32* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_mat22d(mat22d& a, float64* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_mat33d(mat33d& a, float64* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_mat44d(mat44d& a, float64* ret) { *ret = wp::trace(a); }
WP_API void builtin_trace_spatial_matrixd(spatial_matrixd& a, float64* ret) { *ret = wp::trace(a); }
WP_API void builtin_diag_vec2h(vec2h& vec, mat22h* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_vec3h(vec3h& vec, mat33h* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_vec4h(vec4h& vec, mat44h* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_spatial_vectorh(spatial_vectorh& vec, spatial_matrixh* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_vec2f(vec2f& vec, mat22f* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_vec3f(vec3f& vec, mat33f* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_vec4f(vec4f& vec, mat44f* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_spatial_vectorf(spatial_vectorf& vec, spatial_matrixf* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_vec2d(vec2d& vec, mat22d* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_vec3d(vec3d& vec, mat33d* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_vec4d(vec4d& vec, mat44d* ret) { *ret = wp::diag(vec); }
WP_API void builtin_diag_spatial_vectord(spatial_vectord& vec, spatial_matrixd* ret) { *ret = wp::diag(vec); }
WP_API void builtin_get_diag_mat22h(mat22h& mat, vec2h* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_mat33h(mat33h& mat, vec3h* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_mat44h(mat44h& mat, vec4h* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_spatial_matrixh(spatial_matrixh& mat, spatial_vectorh* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_mat22f(mat22f& mat, vec2f* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_mat33f(mat33f& mat, vec3f* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_mat44f(mat44f& mat, vec4f* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_spatial_matrixf(spatial_matrixf& mat, spatial_vectorf* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_mat22d(mat22d& mat, vec2d* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_mat33d(mat33d& mat, vec3d* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_mat44d(mat44d& mat, vec4d* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_get_diag_spatial_matrixd(spatial_matrixd& mat, spatial_vectord* ret) { *ret = wp::get_diag(mat); }
WP_API void builtin_cw_mul_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_mul_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret) { *ret = wp::cw_mul(a, b); }
WP_API void builtin_cw_div_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_cw_div_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret) { *ret = wp::cw_div(a, b); }
WP_API void builtin_quat_identity_float16(quath* ret) { *ret = wp::quat_identity<float16>(); }
WP_API void builtin_quat_identity_float32(quatf* ret) { *ret = wp::quat_identity<float32>(); }
WP_API void builtin_quat_identity_float64(quatd* ret) { *ret = wp::quat_identity<float64>(); }
WP_API void builtin_quat_from_axis_angle_vec3h_float16(vec3h& axis, float16 angle, quath* ret) { *ret = wp::quat_from_axis_angle(axis, angle); }
WP_API void builtin_quat_from_axis_angle_vec3f_float32(vec3f& axis, float32 angle, quatf* ret) { *ret = wp::quat_from_axis_angle(axis, angle); }
WP_API void builtin_quat_from_axis_angle_vec3d_float64(vec3d& axis, float64 angle, quatd* ret) { *ret = wp::quat_from_axis_angle(axis, angle); }
WP_API void builtin_quat_from_matrix_mat33h(mat33h& mat, quath* ret) { *ret = wp::quat_from_matrix(mat); }
WP_API void builtin_quat_from_matrix_mat33f(mat33f& mat, quatf* ret) { *ret = wp::quat_from_matrix(mat); }
WP_API void builtin_quat_from_matrix_mat33d(mat33d& mat, quatd* ret) { *ret = wp::quat_from_matrix(mat); }
WP_API void builtin_quat_rpy_float16_float16_float16(float16 roll, float16 pitch, float16 yaw, quath* ret) { *ret = wp::quat_rpy(roll, pitch, yaw); }
WP_API void builtin_quat_rpy_float32_float32_float32(float32 roll, float32 pitch, float32 yaw, quatf* ret) { *ret = wp::quat_rpy(roll, pitch, yaw); }
WP_API void builtin_quat_rpy_float64_float64_float64(float64 roll, float64 pitch, float64 yaw, quatd* ret) { *ret = wp::quat_rpy(roll, pitch, yaw); }
WP_API void builtin_quat_inverse_quath(quath& quat, quath* ret) { *ret = wp::quat_inverse(quat); }
WP_API void builtin_quat_inverse_quatf(quatf& quat, quatf* ret) { *ret = wp::quat_inverse(quat); }
WP_API void builtin_quat_inverse_quatd(quatd& quat, quatd* ret) { *ret = wp::quat_inverse(quat); }
WP_API void builtin_quat_rotate_quath_vec3h(quath& quat, vec3h& vec, vec3h* ret) { *ret = wp::quat_rotate(quat, vec); }
WP_API void builtin_quat_rotate_quatf_vec3f(quatf& quat, vec3f& vec, vec3f* ret) { *ret = wp::quat_rotate(quat, vec); }
WP_API void builtin_quat_rotate_quatd_vec3d(quatd& quat, vec3d& vec, vec3d* ret) { *ret = wp::quat_rotate(quat, vec); }
WP_API void builtin_quat_rotate_inv_quath_vec3h(quath& quat, vec3h& vec, vec3h* ret) { *ret = wp::quat_rotate_inv(quat, vec); }
WP_API void builtin_quat_rotate_inv_quatf_vec3f(quatf& quat, vec3f& vec, vec3f* ret) { *ret = wp::quat_rotate_inv(quat, vec); }
WP_API void builtin_quat_rotate_inv_quatd_vec3d(quatd& quat, vec3d& vec, vec3d* ret) { *ret = wp::quat_rotate_inv(quat, vec); }
WP_API void builtin_quat_slerp_quath_quath_float16(quath& a, quath& b, float16 t, quath* ret) { *ret = wp::quat_slerp(a, b, t); }
WP_API void builtin_quat_slerp_quatf_quatf_float32(quatf& a, quatf& b, float32 t, quatf* ret) { *ret = wp::quat_slerp(a, b, t); }
WP_API void builtin_quat_slerp_quatd_quatd_float64(quatd& a, quatd& b, float64 t, quatd* ret) { *ret = wp::quat_slerp(a, b, t); }
WP_API void builtin_quat_to_matrix_quath(quath& quat, mat33h* ret) { *ret = wp::quat_to_matrix(quat); }
WP_API void builtin_quat_to_matrix_quatf(quatf& quat, mat33f* ret) { *ret = wp::quat_to_matrix(quat); }
WP_API void builtin_quat_to_matrix_quatd(quatd& quat, mat33d* ret) { *ret = wp::quat_to_matrix(quat); }
WP_API void builtin_transform_identity_float16(transformh* ret) { *ret = wp::transform_identity<float16>(); }
WP_API void builtin_transform_identity_float32(transformf* ret) { *ret = wp::transform_identity<float32>(); }
WP_API void builtin_transform_identity_float64(transformd* ret) { *ret = wp::transform_identity<float64>(); }
WP_API void builtin_transform_get_translation_transformh(transformh& xform, vec3h* ret) { *ret = wp::transform_get_translation(xform); }
WP_API void builtin_transform_get_translation_transformf(transformf& xform, vec3f* ret) { *ret = wp::transform_get_translation(xform); }
WP_API void builtin_transform_get_translation_transformd(transformd& xform, vec3d* ret) { *ret = wp::transform_get_translation(xform); }
WP_API void builtin_transform_get_rotation_transformh(transformh& xform, quath* ret) { *ret = wp::transform_get_rotation(xform); }
WP_API void builtin_transform_get_rotation_transformf(transformf& xform, quatf* ret) { *ret = wp::transform_get_rotation(xform); }
WP_API void builtin_transform_get_rotation_transformd(transformd& xform, quatd* ret) { *ret = wp::transform_get_rotation(xform); }
WP_API void builtin_transform_multiply_transformh_transformh(transformh& a, transformh& b, transformh* ret) { *ret = wp::transform_multiply(a, b); }
WP_API void builtin_transform_multiply_transformf_transformf(transformf& a, transformf& b, transformf* ret) { *ret = wp::transform_multiply(a, b); }
WP_API void builtin_transform_multiply_transformd_transformd(transformd& a, transformd& b, transformd* ret) { *ret = wp::transform_multiply(a, b); }
WP_API void builtin_transform_point_transformh_vec3h(transformh& xform, vec3h& point, vec3h* ret) { *ret = wp::transform_point(xform, point); }
WP_API void builtin_transform_point_transformf_vec3f(transformf& xform, vec3f& point, vec3f* ret) { *ret = wp::transform_point(xform, point); }
WP_API void builtin_transform_point_transformd_vec3d(transformd& xform, vec3d& point, vec3d* ret) { *ret = wp::transform_point(xform, point); }
WP_API void builtin_transform_point_mat44h_vec3h(mat44h& mat, vec3h& point, vec3h* ret) { *ret = wp::transform_point(mat, point); }
WP_API void builtin_transform_point_mat44f_vec3f(mat44f& mat, vec3f& point, vec3f* ret) { *ret = wp::transform_point(mat, point); }
WP_API void builtin_transform_point_mat44d_vec3d(mat44d& mat, vec3d& point, vec3d* ret) { *ret = wp::transform_point(mat, point); }
WP_API void builtin_transform_vector_transformh_vec3h(transformh& xform, vec3h& vec, vec3h* ret) { *ret = wp::transform_vector(xform, vec); }
WP_API void builtin_transform_vector_transformf_vec3f(transformf& xform, vec3f& vec, vec3f* ret) { *ret = wp::transform_vector(xform, vec); }
WP_API void builtin_transform_vector_transformd_vec3d(transformd& xform, vec3d& vec, vec3d* ret) { *ret = wp::transform_vector(xform, vec); }
WP_API void builtin_transform_vector_mat44h_vec3h(mat44h& mat, vec3h& vec, vec3h* ret) { *ret = wp::transform_vector(mat, vec); }
WP_API void builtin_transform_vector_mat44f_vec3f(mat44f& mat, vec3f& vec, vec3f* ret) { *ret = wp::transform_vector(mat, vec); }
WP_API void builtin_transform_vector_mat44d_vec3d(mat44d& mat, vec3d& vec, vec3d* ret) { *ret = wp::transform_vector(mat, vec); }
WP_API void builtin_transform_inverse_transformh(transformh& xform, transformh* ret) { *ret = wp::transform_inverse(xform); }
WP_API void builtin_transform_inverse_transformf(transformf& xform, transformf* ret) { *ret = wp::transform_inverse(xform); }
WP_API void builtin_transform_inverse_transformd(transformd& xform, transformd* ret) { *ret = wp::transform_inverse(xform); }
WP_API void builtin_spatial_dot_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, float16* ret) { *ret = wp::spatial_dot(a, b); }
WP_API void builtin_spatial_dot_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, float32* ret) { *ret = wp::spatial_dot(a, b); }
WP_API void builtin_spatial_dot_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, float64* ret) { *ret = wp::spatial_dot(a, b); }
WP_API void builtin_spatial_cross_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::spatial_cross(a, b); }
WP_API void builtin_spatial_cross_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::spatial_cross(a, b); }
WP_API void builtin_spatial_cross_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::spatial_cross(a, b); }
WP_API void builtin_spatial_cross_dual_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::spatial_cross_dual(a, b); }
WP_API void builtin_spatial_cross_dual_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::spatial_cross_dual(a, b); }
WP_API void builtin_spatial_cross_dual_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::spatial_cross_dual(a, b); }
WP_API void builtin_spatial_top_spatial_vectorh(spatial_vectorh& svec, vec3h* ret) { *ret = wp::spatial_top(svec); }
WP_API void builtin_spatial_top_spatial_vectorf(spatial_vectorf& svec, vec3f* ret) { *ret = wp::spatial_top(svec); }
WP_API void builtin_spatial_top_spatial_vectord(spatial_vectord& svec, vec3d* ret) { *ret = wp::spatial_top(svec); }
WP_API void builtin_spatial_bottom_spatial_vectorh(spatial_vectorh& svec, vec3h* ret) { *ret = wp::spatial_bottom(svec); }
WP_API void builtin_spatial_bottom_spatial_vectorf(spatial_vectorf& svec, vec3f* ret) { *ret = wp::spatial_bottom(svec); }
WP_API void builtin_spatial_bottom_spatial_vectord(spatial_vectord& svec, vec3d* ret) { *ret = wp::spatial_bottom(svec); }
WP_API void builtin_volume_sample_f_uint64_vec3f_int32(uint64 id, vec3f& uvw, int32 sampling_mode, float* ret) { *ret = wp::volume_sample_f(id, uvw, sampling_mode); }
WP_API void builtin_volume_sample_grad_f_uint64_vec3f_int32_vec3f(uint64 id, vec3f& uvw, int32 sampling_mode, vec3f& grad, float* ret) { *ret = wp::volume_sample_grad_f(id, uvw, sampling_mode, grad); }
WP_API void builtin_volume_lookup_f_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, float* ret) { *ret = wp::volume_lookup_f(id, i, j, k); }
WP_API void builtin_volume_sample_v_uint64_vec3f_int32(uint64 id, vec3f& uvw, int32 sampling_mode, vec3f* ret) { *ret = wp::volume_sample_v(id, uvw, sampling_mode); }
WP_API void builtin_volume_lookup_v_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, vec3f* ret) { *ret = wp::volume_lookup_v(id, i, j, k); }
WP_API void builtin_volume_sample_i_uint64_vec3f(uint64 id, vec3f& uvw, int* ret) { *ret = wp::volume_sample_i(id, uvw); }
WP_API void builtin_volume_lookup_i_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, int* ret) { *ret = wp::volume_lookup_i(id, i, j, k); }
WP_API void builtin_volume_lookup_index_uint64_int32_int32_int32(uint64 id, int32 i, int32 j, int32 k, int32* ret) { *ret = wp::volume_lookup_index(id, i, j, k); }
WP_API void builtin_volume_index_to_world_uint64_vec3f(uint64 id, vec3f& uvw, vec3f* ret) { *ret = wp::volume_index_to_world(id, uvw); }
WP_API void builtin_volume_world_to_index_uint64_vec3f(uint64 id, vec3f& xyz, vec3f* ret) { *ret = wp::volume_world_to_index(id, xyz); }
WP_API void builtin_volume_index_to_world_dir_uint64_vec3f(uint64 id, vec3f& uvw, vec3f* ret) { *ret = wp::volume_index_to_world_dir(id, uvw); }
WP_API void builtin_volume_world_to_index_dir_uint64_vec3f(uint64 id, vec3f& xyz, vec3f* ret) { *ret = wp::volume_world_to_index_dir(id, xyz); }
WP_API void builtin_rand_init_int32(int32 seed, uint32* ret) { *ret = wp::rand_init(seed); }
WP_API void builtin_rand_init_int32_int32(int32 seed, int32 offset, uint32* ret) { *ret = wp::rand_init(seed, offset); }
WP_API void builtin_randi_uint32(uint32 state, int* ret) { *ret = wp::randi(state); }
WP_API void builtin_randi_uint32_int32_int32(uint32 state, int32 low, int32 high, int* ret) { *ret = wp::randi(state, low, high); }
WP_API void builtin_randf_uint32(uint32 state, float* ret) { *ret = wp::randf(state); }
WP_API void builtin_randf_uint32_float32_float32(uint32 state, float32 low, float32 high, float* ret) { *ret = wp::randf(state, low, high); }
WP_API void builtin_randn_uint32(uint32 state, float* ret) { *ret = wp::randn(state); }
WP_API void builtin_sample_triangle_uint32(uint32 state, vec2f* ret) { *ret = wp::sample_triangle(state); }
WP_API void builtin_sample_unit_ring_uint32(uint32 state, vec2f* ret) { *ret = wp::sample_unit_ring(state); }
WP_API void builtin_sample_unit_disk_uint32(uint32 state, vec2f* ret) { *ret = wp::sample_unit_disk(state); }
WP_API void builtin_sample_unit_sphere_surface_uint32(uint32 state, vec3f* ret) { *ret = wp::sample_unit_sphere_surface(state); }
WP_API void builtin_sample_unit_sphere_uint32(uint32 state, vec3f* ret) { *ret = wp::sample_unit_sphere(state); }
WP_API void builtin_sample_unit_hemisphere_surface_uint32(uint32 state, vec3f* ret) { *ret = wp::sample_unit_hemisphere_surface(state); }
WP_API void builtin_sample_unit_hemisphere_uint32(uint32 state, vec3f* ret) { *ret = wp::sample_unit_hemisphere(state); }
WP_API void builtin_sample_unit_square_uint32(uint32 state, vec2f* ret) { *ret = wp::sample_unit_square(state); }
WP_API void builtin_sample_unit_cube_uint32(uint32 state, vec3f* ret) { *ret = wp::sample_unit_cube(state); }
WP_API void builtin_poisson_uint32_float32(uint32 state, float32 lam, uint32* ret) { *ret = wp::poisson(state, lam); }
WP_API void builtin_noise_uint32_float32(uint32 state, float32 x, float* ret) { *ret = wp::noise(state, x); }
WP_API void builtin_noise_uint32_vec2f(uint32 state, vec2f& xy, float* ret) { *ret = wp::noise(state, xy); }
WP_API void builtin_noise_uint32_vec3f(uint32 state, vec3f& xyz, float* ret) { *ret = wp::noise(state, xyz); }
WP_API void builtin_noise_uint32_vec4f(uint32 state, vec4f& xyzt, float* ret) { *ret = wp::noise(state, xyzt); }
WP_API void builtin_pnoise_uint32_float32_int32(uint32 state, float32 x, int32 px, float* ret) { *ret = wp::pnoise(state, x, px); }
WP_API void builtin_pnoise_uint32_vec2f_int32_int32(uint32 state, vec2f& xy, int32 px, int32 py, float* ret) { *ret = wp::pnoise(state, xy, px, py); }
WP_API void builtin_pnoise_uint32_vec3f_int32_int32_int32(uint32 state, vec3f& xyz, int32 px, int32 py, int32 pz, float* ret) { *ret = wp::pnoise(state, xyz, px, py, pz); }
WP_API void builtin_pnoise_uint32_vec4f_int32_int32_int32_int32(uint32 state, vec4f& xyzt, int32 px, int32 py, int32 pz, int32 pt, float* ret) { *ret = wp::pnoise(state, xyzt, px, py, pz, pt); }
WP_API void builtin_curlnoise_uint32_vec2f_uint32_float32_float32(uint32 state, vec2f& xy, uint32 octaves, float32 lacunarity, float32 gain, vec2f* ret) { *ret = wp::curlnoise(state, xy, octaves, lacunarity, gain); }
WP_API void builtin_curlnoise_uint32_vec3f_uint32_float32_float32(uint32 state, vec3f& xyz, uint32 octaves, float32 lacunarity, float32 gain, vec3f* ret) { *ret = wp::curlnoise(state, xyz, octaves, lacunarity, gain); }
WP_API void builtin_curlnoise_uint32_vec4f_uint32_float32_float32(uint32 state, vec4f& xyzt, uint32 octaves, float32 lacunarity, float32 gain, vec3f* ret) { *ret = wp::curlnoise(state, xyzt, octaves, lacunarity, gain); }
WP_API void builtin_assign_vec2h_int32_float16(vec2h& a, int32 i, float16 value, vec2h* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3h_int32_float16(vec3h& a, int32 i, float16 value, vec3h* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4h_int32_float16(vec4h& a, int32 i, float16 value, vec4h* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_spatial_vectorh_int32_float16(spatial_vectorh& a, int32 i, float16 value, spatial_vectorh* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2f_int32_float32(vec2f& a, int32 i, float32 value, vec2f* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3f_int32_float32(vec3f& a, int32 i, float32 value, vec3f* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4f_int32_float32(vec4f& a, int32 i, float32 value, vec4f* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_spatial_vectorf_int32_float32(spatial_vectorf& a, int32 i, float32 value, spatial_vectorf* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2d_int32_float64(vec2d& a, int32 i, float64 value, vec2d* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3d_int32_float64(vec3d& a, int32 i, float64 value, vec3d* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4d_int32_float64(vec4d& a, int32 i, float64 value, vec4d* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_spatial_vectord_int32_float64(spatial_vectord& a, int32 i, float64 value, spatial_vectord* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2s_int32_int16(vec2s& a, int32 i, int16 value, vec2s* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3s_int32_int16(vec3s& a, int32 i, int16 value, vec3s* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4s_int32_int16(vec4s& a, int32 i, int16 value, vec4s* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2i_int32_int32(vec2i& a, int32 i, int32 value, vec2i* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3i_int32_int32(vec3i& a, int32 i, int32 value, vec3i* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4i_int32_int32(vec4i& a, int32 i, int32 value, vec4i* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2l_int32_int64(vec2l& a, int32 i, int64 value, vec2l* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3l_int32_int64(vec3l& a, int32 i, int64 value, vec3l* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4l_int32_int64(vec4l& a, int32 i, int64 value, vec4l* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2b_int32_int8(vec2b& a, int32 i, int8 value, vec2b* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3b_int32_int8(vec3b& a, int32 i, int8 value, vec3b* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4b_int32_int8(vec4b& a, int32 i, int8 value, vec4b* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2us_int32_uint16(vec2us& a, int32 i, uint16 value, vec2us* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3us_int32_uint16(vec3us& a, int32 i, uint16 value, vec3us* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4us_int32_uint16(vec4us& a, int32 i, uint16 value, vec4us* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2ui_int32_uint32(vec2ui& a, int32 i, uint32 value, vec2ui* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3ui_int32_uint32(vec3ui& a, int32 i, uint32 value, vec3ui* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4ui_int32_uint32(vec4ui& a, int32 i, uint32 value, vec4ui* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2ul_int32_uint64(vec2ul& a, int32 i, uint64 value, vec2ul* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3ul_int32_uint64(vec3ul& a, int32 i, uint64 value, vec3ul* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4ul_int32_uint64(vec4ul& a, int32 i, uint64 value, vec4ul* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec2ub_int32_uint8(vec2ub& a, int32 i, uint8 value, vec2ub* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec3ub_int32_uint8(vec3ub& a, int32 i, uint8 value, vec3ub* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_vec4ub_int32_uint8(vec4ub& a, int32 i, uint8 value, vec4ub* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_quath_int32_float16(quath& a, int32 i, float16 value, quath* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_quatf_int32_float32(quatf& a, int32 i, float32 value, quatf* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_quatd_int32_float64(quatd& a, int32 i, float64 value, quatd* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat22h_int32_int32_float16(mat22h& a, int32 i, int32 j, float16 value, mat22h* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat33h_int32_int32_float16(mat33h& a, int32 i, int32 j, float16 value, mat33h* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat44h_int32_int32_float16(mat44h& a, int32 i, int32 j, float16 value, mat44h* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_spatial_matrixh_int32_int32_float16(spatial_matrixh& a, int32 i, int32 j, float16 value, spatial_matrixh* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat22f_int32_int32_float32(mat22f& a, int32 i, int32 j, float32 value, mat22f* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat33f_int32_int32_float32(mat33f& a, int32 i, int32 j, float32 value, mat33f* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat44f_int32_int32_float32(mat44f& a, int32 i, int32 j, float32 value, mat44f* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_spatial_matrixf_int32_int32_float32(spatial_matrixf& a, int32 i, int32 j, float32 value, spatial_matrixf* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat22d_int32_int32_float64(mat22d& a, int32 i, int32 j, float64 value, mat22d* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat33d_int32_int32_float64(mat33d& a, int32 i, int32 j, float64 value, mat33d* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat44d_int32_int32_float64(mat44d& a, int32 i, int32 j, float64 value, mat44d* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_spatial_matrixd_int32_int32_float64(spatial_matrixd& a, int32 i, int32 j, float64 value, spatial_matrixd* ret) { *ret = wp::assign(a, i, j, value); }
WP_API void builtin_assign_mat22h_int32_vec2h(mat22h& a, int32 i, vec2h& value, mat22h* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat33h_int32_vec3h(mat33h& a, int32 i, vec3h& value, mat33h* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat44h_int32_vec4h(mat44h& a, int32 i, vec4h& value, mat44h* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_spatial_matrixh_int32_spatial_vectorh(spatial_matrixh& a, int32 i, spatial_vectorh& value, spatial_matrixh* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat22f_int32_vec2f(mat22f& a, int32 i, vec2f& value, mat22f* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat33f_int32_vec3f(mat33f& a, int32 i, vec3f& value, mat33f* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat44f_int32_vec4f(mat44f& a, int32 i, vec4f& value, mat44f* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_spatial_matrixf_int32_spatial_vectorf(spatial_matrixf& a, int32 i, spatial_vectorf& value, spatial_matrixf* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat22d_int32_vec2d(mat22d& a, int32 i, vec2d& value, mat22d* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat33d_int32_vec3d(mat33d& a, int32 i, vec3d& value, mat33d* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_mat44d_int32_vec4d(mat44d& a, int32 i, vec4d& value, mat44d* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_assign_spatial_matrixd_int32_spatial_vectord(spatial_matrixd& a, int32 i, spatial_vectord& value, spatial_matrixd* ret) { *ret = wp::assign(a, i, value); }
WP_API void builtin_extract_vec2h_int32(vec2h& a, int32 i, float16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3h_int32(vec3h& a, int32 i, float16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4h_int32(vec4h& a, int32 i, float16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_spatial_vectorh_int32(spatial_vectorh& a, int32 i, float16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2f_int32(vec2f& a, int32 i, float32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3f_int32(vec3f& a, int32 i, float32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4f_int32(vec4f& a, int32 i, float32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_spatial_vectorf_int32(spatial_vectorf& a, int32 i, float32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2d_int32(vec2d& a, int32 i, float64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3d_int32(vec3d& a, int32 i, float64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4d_int32(vec4d& a, int32 i, float64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_spatial_vectord_int32(spatial_vectord& a, int32 i, float64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2s_int32(vec2s& a, int32 i, int16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3s_int32(vec3s& a, int32 i, int16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4s_int32(vec4s& a, int32 i, int16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2i_int32(vec2i& a, int32 i, int32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3i_int32(vec3i& a, int32 i, int32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4i_int32(vec4i& a, int32 i, int32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2l_int32(vec2l& a, int32 i, int64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3l_int32(vec3l& a, int32 i, int64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4l_int32(vec4l& a, int32 i, int64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2b_int32(vec2b& a, int32 i, int8* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3b_int32(vec3b& a, int32 i, int8* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4b_int32(vec4b& a, int32 i, int8* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2us_int32(vec2us& a, int32 i, uint16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3us_int32(vec3us& a, int32 i, uint16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4us_int32(vec4us& a, int32 i, uint16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2ui_int32(vec2ui& a, int32 i, uint32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3ui_int32(vec3ui& a, int32 i, uint32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4ui_int32(vec4ui& a, int32 i, uint32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2ul_int32(vec2ul& a, int32 i, uint64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3ul_int32(vec3ul& a, int32 i, uint64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4ul_int32(vec4ul& a, int32 i, uint64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec2ub_int32(vec2ub& a, int32 i, uint8* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec3ub_int32(vec3ub& a, int32 i, uint8* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_vec4ub_int32(vec4ub& a, int32 i, uint8* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_quath_int32(quath& a, int32 i, float16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_quatf_int32(quatf& a, int32 i, float32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_quatd_int32(quatd& a, int32 i, float64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat22h_int32(mat22h& a, int32 i, vec2h* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat33h_int32(mat33h& a, int32 i, vec3h* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat44h_int32(mat44h& a, int32 i, vec4h* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_spatial_matrixh_int32(spatial_matrixh& a, int32 i, spatial_vectorh* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat22f_int32(mat22f& a, int32 i, vec2f* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat33f_int32(mat33f& a, int32 i, vec3f* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat44f_int32(mat44f& a, int32 i, vec4f* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_spatial_matrixf_int32(spatial_matrixf& a, int32 i, spatial_vectorf* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat22d_int32(mat22d& a, int32 i, vec2d* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat33d_int32(mat33d& a, int32 i, vec3d* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat44d_int32(mat44d& a, int32 i, vec4d* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_spatial_matrixd_int32(spatial_matrixd& a, int32 i, spatial_vectord* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_mat22h_int32_int32(mat22h& a, int32 i, int32 j, float16* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_mat33h_int32_int32(mat33h& a, int32 i, int32 j, float16* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_mat44h_int32_int32(mat44h& a, int32 i, int32 j, float16* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_spatial_matrixh_int32_int32(spatial_matrixh& a, int32 i, int32 j, float16* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_mat22f_int32_int32(mat22f& a, int32 i, int32 j, float32* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_mat33f_int32_int32(mat33f& a, int32 i, int32 j, float32* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_mat44f_int32_int32(mat44f& a, int32 i, int32 j, float32* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_spatial_matrixf_int32_int32(spatial_matrixf& a, int32 i, int32 j, float32* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_mat22d_int32_int32(mat22d& a, int32 i, int32 j, float64* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_mat33d_int32_int32(mat33d& a, int32 i, int32 j, float64* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_mat44d_int32_int32(mat44d& a, int32 i, int32 j, float64* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_spatial_matrixd_int32_int32(spatial_matrixd& a, int32 i, int32 j, float64* ret) { *ret = wp::extract(a, i, j); }
WP_API void builtin_extract_transformh_int32(transformh& a, int32 i, float16* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_transformf_int32(transformf& a, int32 i, float32* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_transformd_int32(transformd& a, int32 i, float64* ret) { *ret = wp::extract(a, i); }
WP_API void builtin_extract_shape_t_int32(shape_t s, int32 i, int* ret) { *ret = wp::extract(s, i); }
WP_API void builtin_lerp_float16_float16_float16(float16 a, float16 b, float16 t, float16* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_float32_float32_float32(float32 a, float32 b, float32 t, float32* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_float64_float64_float64(float64 a, float64 b, float64 t, float64* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec2h_vec2h_float16(vec2h& a, vec2h& b, float16 t, vec2h* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec3h_vec3h_float16(vec3h& a, vec3h& b, float16 t, vec3h* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec4h_vec4h_float16(vec4h& a, vec4h& b, float16 t, vec4h* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_spatial_vectorh_spatial_vectorh_float16(spatial_vectorh& a, spatial_vectorh& b, float16 t, spatial_vectorh* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec2f_vec2f_float32(vec2f& a, vec2f& b, float32 t, vec2f* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec3f_vec3f_float32(vec3f& a, vec3f& b, float32 t, vec3f* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec4f_vec4f_float32(vec4f& a, vec4f& b, float32 t, vec4f* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_spatial_vectorf_spatial_vectorf_float32(spatial_vectorf& a, spatial_vectorf& b, float32 t, spatial_vectorf* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec2d_vec2d_float64(vec2d& a, vec2d& b, float64 t, vec2d* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec3d_vec3d_float64(vec3d& a, vec3d& b, float64 t, vec3d* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_vec4d_vec4d_float64(vec4d& a, vec4d& b, float64 t, vec4d* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_spatial_vectord_spatial_vectord_float64(spatial_vectord& a, spatial_vectord& b, float64 t, spatial_vectord* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat22h_mat22h_float16(mat22h& a, mat22h& b, float16 t, mat22h* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat33h_mat33h_float16(mat33h& a, mat33h& b, float16 t, mat33h* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat44h_mat44h_float16(mat44h& a, mat44h& b, float16 t, mat44h* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_spatial_matrixh_spatial_matrixh_float16(spatial_matrixh& a, spatial_matrixh& b, float16 t, spatial_matrixh* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat22f_mat22f_float32(mat22f& a, mat22f& b, float32 t, mat22f* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat33f_mat33f_float32(mat33f& a, mat33f& b, float32 t, mat33f* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat44f_mat44f_float32(mat44f& a, mat44f& b, float32 t, mat44f* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_spatial_matrixf_spatial_matrixf_float32(spatial_matrixf& a, spatial_matrixf& b, float32 t, spatial_matrixf* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat22d_mat22d_float64(mat22d& a, mat22d& b, float64 t, mat22d* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat33d_mat33d_float64(mat33d& a, mat33d& b, float64 t, mat33d* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_mat44d_mat44d_float64(mat44d& a, mat44d& b, float64 t, mat44d* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_spatial_matrixd_spatial_matrixd_float64(spatial_matrixd& a, spatial_matrixd& b, float64 t, spatial_matrixd* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_quath_quath_float16(quath& a, quath& b, float16 t, quath* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_quatf_quatf_float32(quatf& a, quatf& b, float32 t, quatf* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_quatd_quatd_float64(quatd& a, quatd& b, float64 t, quatd* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_transformh_transformh_float16(transformh& a, transformh& b, float16 t, transformh* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_transformf_transformf_float32(transformf& a, transformf& b, float32 t, transformf* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_lerp_transformd_transformd_float64(transformd& a, transformd& b, float64 t, transformd* ret) { *ret = wp::lerp(a, b, t); }
WP_API void builtin_smoothstep_float16_float16_float16(float16 a, float16 b, float16 x, float16* ret) { *ret = wp::smoothstep(a, b, x); }
WP_API void builtin_smoothstep_float32_float32_float32(float32 a, float32 b, float32 x, float32* ret) { *ret = wp::smoothstep(a, b, x); }
WP_API void builtin_smoothstep_float64_float64_float64(float64 a, float64 b, float64 x, float64* ret) { *ret = wp::smoothstep(a, b, x); }
WP_API void builtin_add_float16_float16(float16 a, float16 b, float16* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_float32_float32(float32 a, float32 b, float32* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_float64_float64(float64 a, float64 b, float64* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_quath_quath(quath& a, quath& b, quath* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_quatf_quatf(quatf& a, quatf& b, quatf* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_quatd_quatd(quatd& a, quatd& b, quatd* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_transformh_transformh(transformh& a, transformh& b, transformh* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_transformf_transformf(transformf& a, transformf& b, transformf* ret) { *ret = wp::add(a, b); }
WP_API void builtin_add_transformd_transformd(transformd& a, transformd& b, transformd* ret) { *ret = wp::add(a, b); }
WP_API void builtin_sub_float16_float16(float16 a, float16 b, float16* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_float32_float32(float32 a, float32 b, float32* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_float64_float64(float64 a, float64 b, float64* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_quath_quath(quath& a, quath& b, quath* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_quatf_quatf(quatf& a, quatf& b, quatf* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_quatd_quatd(quatd& a, quatd& b, quatd* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_transformh_transformh(transformh& a, transformh& b, transformh* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_transformf_transformf(transformf& a, transformf& b, transformf* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_sub_transformd_transformd(transformd& a, transformd& b, transformd* ret) { *ret = wp::sub(a, b); }
WP_API void builtin_bit_and_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::bit_and(a, b); }
WP_API void builtin_bit_and_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::bit_and(a, b); }
WP_API void builtin_bit_and_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::bit_and(a, b); }
WP_API void builtin_bit_and_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::bit_and(a, b); }
WP_API void builtin_bit_and_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::bit_and(a, b); }
WP_API void builtin_bit_and_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::bit_and(a, b); }
WP_API void builtin_bit_and_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::bit_and(a, b); }
WP_API void builtin_bit_and_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::bit_and(a, b); }
WP_API void builtin_bit_or_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::bit_or(a, b); }
WP_API void builtin_bit_or_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::bit_or(a, b); }
WP_API void builtin_bit_or_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::bit_or(a, b); }
WP_API void builtin_bit_or_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::bit_or(a, b); }
WP_API void builtin_bit_or_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::bit_or(a, b); }
WP_API void builtin_bit_or_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::bit_or(a, b); }
WP_API void builtin_bit_or_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::bit_or(a, b); }
WP_API void builtin_bit_or_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::bit_or(a, b); }
WP_API void builtin_bit_xor_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::bit_xor(a, b); }
WP_API void builtin_bit_xor_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::bit_xor(a, b); }
WP_API void builtin_bit_xor_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::bit_xor(a, b); }
WP_API void builtin_bit_xor_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::bit_xor(a, b); }
WP_API void builtin_bit_xor_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::bit_xor(a, b); }
WP_API void builtin_bit_xor_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::bit_xor(a, b); }
WP_API void builtin_bit_xor_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::bit_xor(a, b); }
WP_API void builtin_bit_xor_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::bit_xor(a, b); }
WP_API void builtin_lshift_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::lshift(a, b); }
WP_API void builtin_lshift_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::lshift(a, b); }
WP_API void builtin_lshift_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::lshift(a, b); }
WP_API void builtin_lshift_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::lshift(a, b); }
WP_API void builtin_lshift_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::lshift(a, b); }
WP_API void builtin_lshift_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::lshift(a, b); }
WP_API void builtin_lshift_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::lshift(a, b); }
WP_API void builtin_lshift_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::lshift(a, b); }
WP_API void builtin_rshift_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::rshift(a, b); }
WP_API void builtin_rshift_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::rshift(a, b); }
WP_API void builtin_rshift_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::rshift(a, b); }
WP_API void builtin_rshift_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::rshift(a, b); }
WP_API void builtin_rshift_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::rshift(a, b); }
WP_API void builtin_rshift_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::rshift(a, b); }
WP_API void builtin_rshift_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::rshift(a, b); }
WP_API void builtin_rshift_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::rshift(a, b); }
WP_API void builtin_invert_int16(int16 a, int16* ret) { *ret = wp::invert(a); }
WP_API void builtin_invert_int32(int32 a, int32* ret) { *ret = wp::invert(a); }
WP_API void builtin_invert_int64(int64 a, int64* ret) { *ret = wp::invert(a); }
WP_API void builtin_invert_int8(int8 a, int8* ret) { *ret = wp::invert(a); }
WP_API void builtin_invert_uint16(uint16 a, uint16* ret) { *ret = wp::invert(a); }
WP_API void builtin_invert_uint32(uint32 a, uint32* ret) { *ret = wp::invert(a); }
WP_API void builtin_invert_uint64(uint64 a, uint64* ret) { *ret = wp::invert(a); }
WP_API void builtin_invert_uint8(uint8 a, uint8* ret) { *ret = wp::invert(a); }
WP_API void builtin_mul_float16_float16(float16 a, float16 b, float16* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_float32(float32 a, float32 b, float32* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_float64(float64 a, float64 b, float64* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2h_float16(vec2h& a, float16 b, vec2h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3h_float16(vec3h& a, float16 b, vec3h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4h_float16(vec4h& a, float16 b, vec4h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_vectorh_float16(spatial_vectorh& a, float16 b, spatial_vectorh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2f_float32(vec2f& a, float32 b, vec2f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3f_float32(vec3f& a, float32 b, vec3f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4f_float32(vec4f& a, float32 b, vec4f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_vectorf_float32(spatial_vectorf& a, float32 b, spatial_vectorf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2d_float64(vec2d& a, float64 b, vec2d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3d_float64(vec3d& a, float64 b, vec3d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4d_float64(vec4d& a, float64 b, vec4d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_vectord_float64(spatial_vectord& a, float64 b, spatial_vectord* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2s_int16(vec2s& a, int16 b, vec2s* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3s_int16(vec3s& a, int16 b, vec3s* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4s_int16(vec4s& a, int16 b, vec4s* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2i_int32(vec2i& a, int32 b, vec2i* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3i_int32(vec3i& a, int32 b, vec3i* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4i_int32(vec4i& a, int32 b, vec4i* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2l_int64(vec2l& a, int64 b, vec2l* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3l_int64(vec3l& a, int64 b, vec3l* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4l_int64(vec4l& a, int64 b, vec4l* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2b_int8(vec2b& a, int8 b, vec2b* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3b_int8(vec3b& a, int8 b, vec3b* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4b_int8(vec4b& a, int8 b, vec4b* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2us_uint16(vec2us& a, uint16 b, vec2us* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3us_uint16(vec3us& a, uint16 b, vec3us* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4us_uint16(vec4us& a, uint16 b, vec4us* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2ui_uint32(vec2ui& a, uint32 b, vec2ui* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3ui_uint32(vec3ui& a, uint32 b, vec3ui* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4ui_uint32(vec4ui& a, uint32 b, vec4ui* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2ul_uint64(vec2ul& a, uint64 b, vec2ul* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3ul_uint64(vec3ul& a, uint64 b, vec3ul* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4ul_uint64(vec4ul& a, uint64 b, vec4ul* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2ub_uint8(vec2ub& a, uint8 b, vec2ub* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3ub_uint8(vec3ub& a, uint8 b, vec3ub* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4ub_uint8(vec4ub& a, uint8 b, vec4ub* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_vec2h(float16 a, vec2h& b, vec2h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_vec3h(float16 a, vec3h& b, vec3h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_vec4h(float16 a, vec4h& b, vec4h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_spatial_vectorh(float16 a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_vec2f(float32 a, vec2f& b, vec2f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_vec3f(float32 a, vec3f& b, vec3f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_vec4f(float32 a, vec4f& b, vec4f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_spatial_vectorf(float32 a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_vec2d(float64 a, vec2d& b, vec2d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_vec3d(float64 a, vec3d& b, vec3d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_vec4d(float64 a, vec4d& b, vec4d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_spatial_vectord(float64 a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int16_vec2s(int16 a, vec2s& b, vec2s* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int16_vec3s(int16 a, vec3s& b, vec3s* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int16_vec4s(int16 a, vec4s& b, vec4s* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int32_vec2i(int32 a, vec2i& b, vec2i* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int32_vec3i(int32 a, vec3i& b, vec3i* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int32_vec4i(int32 a, vec4i& b, vec4i* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int64_vec2l(int64 a, vec2l& b, vec2l* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int64_vec3l(int64 a, vec3l& b, vec3l* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int64_vec4l(int64 a, vec4l& b, vec4l* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int8_vec2b(int8 a, vec2b& b, vec2b* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int8_vec3b(int8 a, vec3b& b, vec3b* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_int8_vec4b(int8 a, vec4b& b, vec4b* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint16_vec2us(uint16 a, vec2us& b, vec2us* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint16_vec3us(uint16 a, vec3us& b, vec3us* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint16_vec4us(uint16 a, vec4us& b, vec4us* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint32_vec2ui(uint32 a, vec2ui& b, vec2ui* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint32_vec3ui(uint32 a, vec3ui& b, vec3ui* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint32_vec4ui(uint32 a, vec4ui& b, vec4ui* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint64_vec2ul(uint64 a, vec2ul& b, vec2ul* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint64_vec3ul(uint64 a, vec3ul& b, vec3ul* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint64_vec4ul(uint64 a, vec4ul& b, vec4ul* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint8_vec2ub(uint8 a, vec2ub& b, vec2ub* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint8_vec3ub(uint8 a, vec3ub& b, vec3ub* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_uint8_vec4ub(uint8 a, vec4ub& b, vec4ub* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_quath_float16(quath& a, float16 b, quath* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_quatf_float32(quatf& a, float32 b, quatf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_quatd_float64(quatd& a, float64 b, quatd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_quath(float16 a, quath& b, quath* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_quatf(float32 a, quatf& b, quatf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_quatd(float64 a, quatd& b, quatd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_quath_quath(quath& a, quath& b, quath* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_quatf_quatf(quatf& a, quatf& b, quatf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_quatd_quatd(quatd& a, quatd& b, quatd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_mat22h(float16 a, mat22h& b, mat22h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_mat33h(float16 a, mat33h& b, mat33h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_mat44h(float16 a, mat44h& b, mat44h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_spatial_matrixh(float16 a, spatial_matrixh& b, spatial_matrixh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_mat22f(float32 a, mat22f& b, mat22f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_mat33f(float32 a, mat33f& b, mat33f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_mat44f(float32 a, mat44f& b, mat44f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_spatial_matrixf(float32 a, spatial_matrixf& b, spatial_matrixf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_mat22d(float64 a, mat22d& b, mat22d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_mat33d(float64 a, mat33d& b, mat33d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_mat44d(float64 a, mat44d& b, mat44d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_spatial_matrixd(float64 a, spatial_matrixd& b, spatial_matrixd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22h_float16(mat22h& a, float16 b, mat22h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33h_float16(mat33h& a, float16 b, mat33h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44h_float16(mat44h& a, float16 b, mat44h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixh_float16(spatial_matrixh& a, float16 b, spatial_matrixh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22f_float32(mat22f& a, float32 b, mat22f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33f_float32(mat33f& a, float32 b, mat33f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44f_float32(mat44f& a, float32 b, mat44f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixf_float32(spatial_matrixf& a, float32 b, spatial_matrixf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22d_float64(mat22d& a, float64 b, mat22d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33d_float64(mat33d& a, float64 b, mat33d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44d_float64(mat44d& a, float64 b, mat44d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixd_float64(spatial_matrixd& a, float64 b, spatial_matrixd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22h_vec2h(mat22h& a, vec2h& b, vec2h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33h_vec3h(mat33h& a, vec3h& b, vec3h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44h_vec4h(mat44h& a, vec4h& b, vec4h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixh_spatial_vectorh(spatial_matrixh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22f_vec2f(mat22f& a, vec2f& b, vec2f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33f_vec3f(mat33f& a, vec3f& b, vec3f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44f_vec4f(mat44f& a, vec4f& b, vec4f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixf_spatial_vectorf(spatial_matrixf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22d_vec2d(mat22d& a, vec2d& b, vec2d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33d_vec3d(mat33d& a, vec3d& b, vec3d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44d_vec4d(mat44d& a, vec4d& b, vec4d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixd_spatial_vectord(spatial_matrixd& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2h_mat22h(vec2h& a, mat22h& b, vec2h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3h_mat33h(vec3h& a, mat33h& b, vec3h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4h_mat44h(vec4h& a, mat44h& b, vec4h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_vectorh_spatial_matrixh(spatial_vectorh& a, spatial_matrixh& b, spatial_vectorh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2f_mat22f(vec2f& a, mat22f& b, vec2f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3f_mat33f(vec3f& a, mat33f& b, vec3f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4f_mat44f(vec4f& a, mat44f& b, vec4f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_vectorf_spatial_matrixf(spatial_vectorf& a, spatial_matrixf& b, spatial_vectorf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec2d_mat22d(vec2d& a, mat22d& b, vec2d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec3d_mat33d(vec3d& a, mat33d& b, vec3d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_vec4d_mat44d(vec4d& a, mat44d& b, vec4d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_vectord_spatial_matrixd(spatial_vectord& a, spatial_matrixd& b, spatial_vectord* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22h_mat22h(mat22h& a, mat22h& b, mat22h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33h_mat33h(mat33h& a, mat33h& b, mat33h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44h_mat44h(mat44h& a, mat44h& b, mat44h* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixh_spatial_matrixh(spatial_matrixh& a, spatial_matrixh& b, spatial_matrixh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22f_mat22f(mat22f& a, mat22f& b, mat22f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33f_mat33f(mat33f& a, mat33f& b, mat33f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44f_mat44f(mat44f& a, mat44f& b, mat44f* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixf_spatial_matrixf(spatial_matrixf& a, spatial_matrixf& b, spatial_matrixf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat22d_mat22d(mat22d& a, mat22d& b, mat22d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat33d_mat33d(mat33d& a, mat33d& b, mat33d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_mat44d_mat44d(mat44d& a, mat44d& b, mat44d* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_spatial_matrixd_spatial_matrixd(spatial_matrixd& a, spatial_matrixd& b, spatial_matrixd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_transformh_transformh(transformh& a, transformh& b, transformh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_transformf_transformf(transformf& a, transformf& b, transformf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_transformd_transformd(transformd& a, transformd& b, transformd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float16_transformh(float16 a, transformh& b, transformh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float32_transformf(float32 a, transformf& b, transformf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_float64_transformd(float64 a, transformd& b, transformd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_transformh_float16(transformh& a, float16 b, transformh* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_transformf_float32(transformf& a, float32 b, transformf* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mul_transformd_float64(transformd& a, float64 b, transformd* ret) { *ret = wp::mul(a, b); }
WP_API void builtin_mod_float16_float16(float16 a, float16 b, float16* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_float32_float32(float32 a, float32 b, float32* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_float64_float64(float64 a, float64 b, float64* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2h_vec2h(vec2h& a, vec2h& b, vec2h* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3h_vec3h(vec3h& a, vec3h& b, vec3h* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4h_vec4h(vec4h& a, vec4h& b, vec4h* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_spatial_vectorh_spatial_vectorh(spatial_vectorh& a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2f_vec2f(vec2f& a, vec2f& b, vec2f* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3f_vec3f(vec3f& a, vec3f& b, vec3f* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4f_vec4f(vec4f& a, vec4f& b, vec4f* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_spatial_vectorf_spatial_vectorf(spatial_vectorf& a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2d_vec2d(vec2d& a, vec2d& b, vec2d* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3d_vec3d(vec3d& a, vec3d& b, vec3d* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4d_vec4d(vec4d& a, vec4d& b, vec4d* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_spatial_vectord_spatial_vectord(spatial_vectord& a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2s_vec2s(vec2s& a, vec2s& b, vec2s* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3s_vec3s(vec3s& a, vec3s& b, vec3s* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4s_vec4s(vec4s& a, vec4s& b, vec4s* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2i_vec2i(vec2i& a, vec2i& b, vec2i* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3i_vec3i(vec3i& a, vec3i& b, vec3i* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4i_vec4i(vec4i& a, vec4i& b, vec4i* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2l_vec2l(vec2l& a, vec2l& b, vec2l* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3l_vec3l(vec3l& a, vec3l& b, vec3l* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4l_vec4l(vec4l& a, vec4l& b, vec4l* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2b_vec2b(vec2b& a, vec2b& b, vec2b* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3b_vec3b(vec3b& a, vec3b& b, vec3b* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4b_vec4b(vec4b& a, vec4b& b, vec4b* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2us_vec2us(vec2us& a, vec2us& b, vec2us* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3us_vec3us(vec3us& a, vec3us& b, vec3us* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4us_vec4us(vec4us& a, vec4us& b, vec4us* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2ui_vec2ui(vec2ui& a, vec2ui& b, vec2ui* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3ui_vec3ui(vec3ui& a, vec3ui& b, vec3ui* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4ui_vec4ui(vec4ui& a, vec4ui& b, vec4ui* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2ul_vec2ul(vec2ul& a, vec2ul& b, vec2ul* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3ul_vec3ul(vec3ul& a, vec3ul& b, vec3ul* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4ul_vec4ul(vec4ul& a, vec4ul& b, vec4ul* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec2ub_vec2ub(vec2ub& a, vec2ub& b, vec2ub* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec3ub_vec3ub(vec3ub& a, vec3ub& b, vec3ub* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_mod_vec4ub_vec4ub(vec4ub& a, vec4ub& b, vec4ub* ret) { *ret = wp::mod(a, b); }
WP_API void builtin_div_float16_float16(float16 a, float16 b, float16* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_float32(float32 a, float32 b, float32* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_float64(float64 a, float64 b, float64* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2h_float16(vec2h& a, float16 b, vec2h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3h_float16(vec3h& a, float16 b, vec3h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4h_float16(vec4h& a, float16 b, vec4h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_spatial_vectorh_float16(spatial_vectorh& a, float16 b, spatial_vectorh* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2f_float32(vec2f& a, float32 b, vec2f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3f_float32(vec3f& a, float32 b, vec3f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4f_float32(vec4f& a, float32 b, vec4f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_spatial_vectorf_float32(spatial_vectorf& a, float32 b, spatial_vectorf* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2d_float64(vec2d& a, float64 b, vec2d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3d_float64(vec3d& a, float64 b, vec3d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4d_float64(vec4d& a, float64 b, vec4d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_spatial_vectord_float64(spatial_vectord& a, float64 b, spatial_vectord* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2s_int16(vec2s& a, int16 b, vec2s* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3s_int16(vec3s& a, int16 b, vec3s* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4s_int16(vec4s& a, int16 b, vec4s* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2i_int32(vec2i& a, int32 b, vec2i* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3i_int32(vec3i& a, int32 b, vec3i* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4i_int32(vec4i& a, int32 b, vec4i* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2l_int64(vec2l& a, int64 b, vec2l* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3l_int64(vec3l& a, int64 b, vec3l* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4l_int64(vec4l& a, int64 b, vec4l* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2b_int8(vec2b& a, int8 b, vec2b* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3b_int8(vec3b& a, int8 b, vec3b* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4b_int8(vec4b& a, int8 b, vec4b* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2us_uint16(vec2us& a, uint16 b, vec2us* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3us_uint16(vec3us& a, uint16 b, vec3us* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4us_uint16(vec4us& a, uint16 b, vec4us* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2ui_uint32(vec2ui& a, uint32 b, vec2ui* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3ui_uint32(vec3ui& a, uint32 b, vec3ui* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4ui_uint32(vec4ui& a, uint32 b, vec4ui* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2ul_uint64(vec2ul& a, uint64 b, vec2ul* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3ul_uint64(vec3ul& a, uint64 b, vec3ul* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4ul_uint64(vec4ul& a, uint64 b, vec4ul* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec2ub_uint8(vec2ub& a, uint8 b, vec2ub* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec3ub_uint8(vec3ub& a, uint8 b, vec3ub* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_vec4ub_uint8(vec4ub& a, uint8 b, vec4ub* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_vec2h(float16 a, vec2h& b, vec2h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_vec3h(float16 a, vec3h& b, vec3h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_vec4h(float16 a, vec4h& b, vec4h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_spatial_vectorh(float16 a, spatial_vectorh& b, spatial_vectorh* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_vec2f(float32 a, vec2f& b, vec2f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_vec3f(float32 a, vec3f& b, vec3f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_vec4f(float32 a, vec4f& b, vec4f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_spatial_vectorf(float32 a, spatial_vectorf& b, spatial_vectorf* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_vec2d(float64 a, vec2d& b, vec2d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_vec3d(float64 a, vec3d& b, vec3d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_vec4d(float64 a, vec4d& b, vec4d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_spatial_vectord(float64 a, spatial_vectord& b, spatial_vectord* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int16_vec2s(int16 a, vec2s& b, vec2s* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int16_vec3s(int16 a, vec3s& b, vec3s* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int16_vec4s(int16 a, vec4s& b, vec4s* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int32_vec2i(int32 a, vec2i& b, vec2i* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int32_vec3i(int32 a, vec3i& b, vec3i* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int32_vec4i(int32 a, vec4i& b, vec4i* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int64_vec2l(int64 a, vec2l& b, vec2l* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int64_vec3l(int64 a, vec3l& b, vec3l* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int64_vec4l(int64 a, vec4l& b, vec4l* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int8_vec2b(int8 a, vec2b& b, vec2b* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int8_vec3b(int8 a, vec3b& b, vec3b* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_int8_vec4b(int8 a, vec4b& b, vec4b* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint16_vec2us(uint16 a, vec2us& b, vec2us* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint16_vec3us(uint16 a, vec3us& b, vec3us* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint16_vec4us(uint16 a, vec4us& b, vec4us* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint32_vec2ui(uint32 a, vec2ui& b, vec2ui* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint32_vec3ui(uint32 a, vec3ui& b, vec3ui* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint32_vec4ui(uint32 a, vec4ui& b, vec4ui* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint64_vec2ul(uint64 a, vec2ul& b, vec2ul* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint64_vec3ul(uint64 a, vec3ul& b, vec3ul* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint64_vec4ul(uint64 a, vec4ul& b, vec4ul* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint8_vec2ub(uint8 a, vec2ub& b, vec2ub* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint8_vec3ub(uint8 a, vec3ub& b, vec3ub* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_uint8_vec4ub(uint8 a, vec4ub& b, vec4ub* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat22h_float16(mat22h& a, float16 b, mat22h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat33h_float16(mat33h& a, float16 b, mat33h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat44h_float16(mat44h& a, float16 b, mat44h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_spatial_matrixh_float16(spatial_matrixh& a, float16 b, spatial_matrixh* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat22f_float32(mat22f& a, float32 b, mat22f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat33f_float32(mat33f& a, float32 b, mat33f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat44f_float32(mat44f& a, float32 b, mat44f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_spatial_matrixf_float32(spatial_matrixf& a, float32 b, spatial_matrixf* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat22d_float64(mat22d& a, float64 b, mat22d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat33d_float64(mat33d& a, float64 b, mat33d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_mat44d_float64(mat44d& a, float64 b, mat44d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_spatial_matrixd_float64(spatial_matrixd& a, float64 b, spatial_matrixd* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_mat22h(float16 a, mat22h& b, mat22h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_mat33h(float16 a, mat33h& b, mat33h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_mat44h(float16 a, mat44h& b, mat44h* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_spatial_matrixh(float16 a, spatial_matrixh& b, spatial_matrixh* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_mat22f(float32 a, mat22f& b, mat22f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_mat33f(float32 a, mat33f& b, mat33f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_mat44f(float32 a, mat44f& b, mat44f* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_spatial_matrixf(float32 a, spatial_matrixf& b, spatial_matrixf* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_mat22d(float64 a, mat22d& b, mat22d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_mat33d(float64 a, mat33d& b, mat33d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_mat44d(float64 a, mat44d& b, mat44d* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_spatial_matrixd(float64 a, spatial_matrixd& b, spatial_matrixd* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_quath_float16(quath& a, float16 b, quath* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_quatf_float32(quatf& a, float32 b, quatf* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_quatd_float64(quatd& a, float64 b, quatd* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float16_quath(float16 a, quath& b, quath* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float32_quatf(float32 a, quatf& b, quatf* ret) { *ret = wp::div(a, b); }
WP_API void builtin_div_float64_quatd(float64 a, quatd& b, quatd* ret) { *ret = wp::div(a, b); }
WP_API void builtin_floordiv_float16_float16(float16 a, float16 b, float16* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_float32_float32(float32 a, float32 b, float32* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_float64_float64(float64 a, float64 b, float64* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_int16_int16(int16 a, int16 b, int16* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_int32_int32(int32 a, int32 b, int32* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_int64_int64(int64 a, int64 b, int64* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_int8_int8(int8 a, int8 b, int8* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_uint16_uint16(uint16 a, uint16 b, uint16* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_uint32_uint32(uint32 a, uint32 b, uint32* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_uint64_uint64(uint64 a, uint64 b, uint64* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_floordiv_uint8_uint8(uint8 a, uint8 b, uint8* ret) { *ret = wp::floordiv(a, b); }
WP_API void builtin_pos_float16(float16 x, float16* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_float32(float32 x, float32* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_float64(float64 x, float64* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_int16(int16 x, int16* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_int32(int32 x, int32* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_int64(int64 x, int64* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_int8(int8 x, int8* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_uint16(uint16 x, uint16* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_uint32(uint32 x, uint32* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_uint64(uint64 x, uint64* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_uint8(uint8 x, uint8* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2h(vec2h& x, vec2h* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3h(vec3h& x, vec3h* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4h(vec4h& x, vec4h* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_spatial_vectorh(spatial_vectorh& x, spatial_vectorh* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2f(vec2f& x, vec2f* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3f(vec3f& x, vec3f* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4f(vec4f& x, vec4f* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_spatial_vectorf(spatial_vectorf& x, spatial_vectorf* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2d(vec2d& x, vec2d* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3d(vec3d& x, vec3d* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4d(vec4d& x, vec4d* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_spatial_vectord(spatial_vectord& x, spatial_vectord* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2s(vec2s& x, vec2s* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3s(vec3s& x, vec3s* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4s(vec4s& x, vec4s* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2i(vec2i& x, vec2i* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3i(vec3i& x, vec3i* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4i(vec4i& x, vec4i* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2l(vec2l& x, vec2l* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3l(vec3l& x, vec3l* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4l(vec4l& x, vec4l* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2b(vec2b& x, vec2b* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3b(vec3b& x, vec3b* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4b(vec4b& x, vec4b* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2us(vec2us& x, vec2us* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3us(vec3us& x, vec3us* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4us(vec4us& x, vec4us* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2ui(vec2ui& x, vec2ui* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3ui(vec3ui& x, vec3ui* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4ui(vec4ui& x, vec4ui* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2ul(vec2ul& x, vec2ul* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3ul(vec3ul& x, vec3ul* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4ul(vec4ul& x, vec4ul* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec2ub(vec2ub& x, vec2ub* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec3ub(vec3ub& x, vec3ub* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_vec4ub(vec4ub& x, vec4ub* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_quath(quath& x, quath* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_quatf(quatf& x, quatf* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_quatd(quatd& x, quatd* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat22h(mat22h& x, mat22h* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat33h(mat33h& x, mat33h* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat44h(mat44h& x, mat44h* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_spatial_matrixh(spatial_matrixh& x, spatial_matrixh* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat22f(mat22f& x, mat22f* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat33f(mat33f& x, mat33f* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat44f(mat44f& x, mat44f* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_spatial_matrixf(spatial_matrixf& x, spatial_matrixf* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat22d(mat22d& x, mat22d* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat33d(mat33d& x, mat33d* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_mat44d(mat44d& x, mat44d* ret) { *ret = wp::pos(x); }
WP_API void builtin_pos_spatial_matrixd(spatial_matrixd& x, spatial_matrixd* ret) { *ret = wp::pos(x); }
WP_API void builtin_neg_float16(float16 x, float16* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_float32(float32 x, float32* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_float64(float64 x, float64* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_int16(int16 x, int16* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_int32(int32 x, int32* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_int64(int64 x, int64* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_int8(int8 x, int8* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_uint16(uint16 x, uint16* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_uint32(uint32 x, uint32* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_uint64(uint64 x, uint64* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_uint8(uint8 x, uint8* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2h(vec2h& x, vec2h* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3h(vec3h& x, vec3h* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4h(vec4h& x, vec4h* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_spatial_vectorh(spatial_vectorh& x, spatial_vectorh* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2f(vec2f& x, vec2f* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3f(vec3f& x, vec3f* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4f(vec4f& x, vec4f* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_spatial_vectorf(spatial_vectorf& x, spatial_vectorf* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2d(vec2d& x, vec2d* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3d(vec3d& x, vec3d* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4d(vec4d& x, vec4d* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_spatial_vectord(spatial_vectord& x, spatial_vectord* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2s(vec2s& x, vec2s* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3s(vec3s& x, vec3s* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4s(vec4s& x, vec4s* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2i(vec2i& x, vec2i* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3i(vec3i& x, vec3i* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4i(vec4i& x, vec4i* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2l(vec2l& x, vec2l* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3l(vec3l& x, vec3l* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4l(vec4l& x, vec4l* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2b(vec2b& x, vec2b* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3b(vec3b& x, vec3b* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4b(vec4b& x, vec4b* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2us(vec2us& x, vec2us* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3us(vec3us& x, vec3us* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4us(vec4us& x, vec4us* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2ui(vec2ui& x, vec2ui* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3ui(vec3ui& x, vec3ui* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4ui(vec4ui& x, vec4ui* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2ul(vec2ul& x, vec2ul* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3ul(vec3ul& x, vec3ul* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4ul(vec4ul& x, vec4ul* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec2ub(vec2ub& x, vec2ub* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec3ub(vec3ub& x, vec3ub* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_vec4ub(vec4ub& x, vec4ub* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_quath(quath& x, quath* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_quatf(quatf& x, quatf* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_quatd(quatd& x, quatd* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat22h(mat22h& x, mat22h* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat33h(mat33h& x, mat33h* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat44h(mat44h& x, mat44h* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_spatial_matrixh(spatial_matrixh& x, spatial_matrixh* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat22f(mat22f& x, mat22f* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat33f(mat33f& x, mat33f* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat44f(mat44f& x, mat44f* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_spatial_matrixf(spatial_matrixf& x, spatial_matrixf* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat22d(mat22d& x, mat22d* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat33d(mat33d& x, mat33d* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_mat44d(mat44d& x, mat44d* ret) { *ret = wp::neg(x); }
WP_API void builtin_neg_spatial_matrixd(spatial_matrixd& x, spatial_matrixd* ret) { *ret = wp::neg(x); }
WP_API void builtin_unot_bool(bool a, bool* ret) { *ret = wp::unot(a); }
WP_API void builtin_unot_int8(int8 a, bool* ret) { *ret = wp::unot(a); }
WP_API void builtin_unot_uint8(uint8 a, bool* ret) { *ret = wp::unot(a); }
WP_API void builtin_unot_int16(int16 a, bool* ret) { *ret = wp::unot(a); }
WP_API void builtin_unot_uint16(uint16 a, bool* ret) { *ret = wp::unot(a); }
WP_API void builtin_unot_int32(int32 a, bool* ret) { *ret = wp::unot(a); }
WP_API void builtin_unot_uint32(uint32 a, bool* ret) { *ret = wp::unot(a); }
WP_API void builtin_unot_int64(int64 a, bool* ret) { *ret = wp::unot(a); }
WP_API void builtin_unot_uint64(uint64 a, bool* ret) { *ret = wp::unot(a); }
}  // namespace wp
