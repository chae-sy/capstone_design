import torch


def float_to_custombit_gpu(value, int_bit, frac_bit):
    # 텐서를 GPU로 이동
    value = value.to("cuda")

    if torch.any((value < -(2**int_bit)) | (value > (2**int_bit))):
        raise ValueError("Value out of bound")

    # Determine sign bit
    sign_bit = torch.where(value < 0, 1, 0)
    value = torch.abs(value)

    # Separate integer and fractional parts
    integer_part = torch.floor(value).long()
    fractional_part = value - integer_part.float()

    # Integer part
    max_integer = (2**int_bit) - 1
    integer_part = torch.where(integer_part > max_integer, max_integer, integer_part)

    # Fractional part
    fractional_part = torch.round(fractional_part * (2**frac_bit)).long()
    max_fractional = (2**frac_bit) - 1
    fractional_part = torch.where(
        fractional_part > max_fractional, max_fractional, fractional_part
    )

    # Pack
    packed_value = (
        (sign_bit << (int_bit + frac_bit))
        | (integer_part << frac_bit)
        | fractional_part
    )
    return packed_value


def custombit_to_float_gpu(packed_value, int_bit, frac_bit):
    # 텐서를 GPU로 이동
    packed_value = packed_value.to("cuda")

    # Extract bits
    total_bits = int_bit + frac_bit + 1
    sign_bit = (packed_value >> (int_bit + frac_bit)) & 0x01
    integer_part = (packed_value >> frac_bit) & ((1 << int_bit) - 1)
    fractional_part = packed_value & ((1 << frac_bit) - 1)

    # Convert to float
    value = integer_part.float() + (fractional_part.float() / (2**frac_bit))
    value = torch.where(sign_bit == 1, -value, value)

    return value


def precision_change_gpu(value, int_bit, frac_bit):
    value1 = float_to_custombit_gpu(value, int_bit, frac_bit)
    value = custombit_to_float_gpu(value1, int_bit, frac_bit)
    return value


def apply_precision_change(tensor, int_bit, frac_bit):
    # GPU로 이동
    tensor = tensor.to("cuda")

    flattened = tensor.reshape(-1)
    transformed_flattened = precision_change_gpu(flattened, int_bit, frac_bit)

    return transformed_flattened.reshape(tensor.shape)


# 모델의 모든 파라미터에 양자화를 적용하는 함수
def apply_custom_quantization_to_model(model, int_bit, frac_bit):
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 각 파라미터에 대해 커스텀 양자화 적용 (apply_precision_change 사용)
            quantized_param = apply_precision_change(param.data, int_bit, frac_bit)
            param.data.copy_(quantized_param)  # 파라미터를 양자화된 값으로 덮어쓰기
            # print(f"{name} 파라미터가 양자화되었습니다.")
