// 放置一些共有的宏
#define DATA_PTR data_ptr


#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)


#define DISPATCH_DOUBLE_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
    switch(TYPEIN)							\
    {									\
    case at::ScalarType::Double:						\
        {									\
    using scalar_t_in = double;					\
    switch(TYPEOUT)							\
        {								\
        case at::ScalarType::Double:					\
        {								\
            using scalar_t_out = double;				\
            __VA_ARGS__;						\
            break;							\
        }								\
        case at::ScalarType::Float:					\
        {								\
            using scalar_t_out = float;				\
            __VA_ARGS__;						\
            break;							\
        }								\
        case at::ScalarType::Half:					\
        {								\
            using scalar_t_out = at::Half;				\
            __VA_ARGS__;						\
            break;							\
        }								\
        case at::ScalarType::BFloat16:				\
        {								\
            using scalar_t_out = at::BFloat16;			\
            __VA_ARGS__;						\
            break;							\
        }								\
        default:							\
        AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
        }								\
    break;								\
        }									\
    case at::ScalarType::Float:						\
        {									\
    using scalar_t_in = float;					\
    switch(TYPEOUT)							\
        {								\
        case at::ScalarType::Float:					\
        {								\
            using scalar_t_out = float;				\
            __VA_ARGS__;						\
            break;							\
        }								\
        case at::ScalarType::Half:					\
        {								\
            using scalar_t_out = at::Half;				\
            __VA_ARGS__;						\
            break;							\
        }								\
        case at::ScalarType::BFloat16:				\
        {								\
            using scalar_t_out = at::BFloat16;			\
            __VA_ARGS__;						\
            break;							\
        }								\
        default:							\
        AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
        }								\
    break;								\
        }									\
    case at::ScalarType::Half:						\
        {									\
    using scalar_t_in = at::Half;					\
    using scalar_t_out = at::Half;					\
    __VA_ARGS__;							\
    break;								\
        }									\
    case at::ScalarType::BFloat16:					\
        {									\
    using scalar_t_in = at::BFloat16;				\
    using scalar_t_out = at::BFloat16;				\
    __VA_ARGS__;							\
    break;								\
        }									\
    default:								\
        AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");	\
    }

 #define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
  switch(TYPEIN)							\
    {									\
    case at::ScalarType::Float:						\
      {									\
	using scalar_t_in = float;					\
	switch(TYPEOUT)							\
	  {								\
	  case at::ScalarType::Float:					\
	    {								\
	      using scalar_t_out = float;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::Half:					\
	    {								\
	      using scalar_t_out = at::Half;				\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  case at::ScalarType::BFloat16:				\
	    {								\
	      using scalar_t_out = at::BFloat16;			\
	      __VA_ARGS__;						\
	      break;							\
	    }								\
	  default:							\
	    AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
	  }								\
	break;								\
      }									\
    case at::ScalarType::Half:						\
      {									\
	using scalar_t_in = at::Half;					\
	using scalar_t_out = at::Half;					\
	__VA_ARGS__;							\
	break;								\
      }									\
    case at::ScalarType::BFloat16:					\
      {									\
	using scalar_t_in = at::BFloat16;				\
	using scalar_t_out = at::BFloat16;				\
	__VA_ARGS__;							\
	break;								\
      }									\
    default:								\
      AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");	\
    }


