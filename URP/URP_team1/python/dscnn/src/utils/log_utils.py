# logging_util.py
import logging
import os  # 로그 파일 존재 여부 확인을 위해 사용

def configure_logging(log_file_path="parameter_log.txt"):
    """로그 설정 및 파일 생성."""
    # 로그 파일이 없으면 생성
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            pass  # 빈 파일 생성

    # 로그 설정 구성
    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_parameter_combination(params, block_number=1, extra_parameters=None):
    logging.info(f"===== START OF PARAMETERS LOG BLOCK {block_number} =====")

    # 조합된 파라미터와 기본 파라미터 모두 기록
    for param_name, param_value in params.items():
        logging.info(f"{param_name}: {param_value}")

    if extra_parameters:
        logging.info("Extra Parameters:")
        for param_name, param_value in extra_parameters.items():
            logging.info(f"{param_name}: {param_value}")

    logging.info(f"===== END OF PARAMETERS LOG BLOCK {block_number} =====\n")
