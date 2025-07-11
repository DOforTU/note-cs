**JC** 

Docker : 애플리케이션과 실행 환경을 컨테이너로 패킹, 어디서든 동일한 환경에서 실행할 수 있게 해주는 경량화된 플랫폼

Docker file : 컨테이너 환경을 정의하는 스크립트 파일

Image : 컨테이너 (프로그램을 열기 위한 환경) 실행을 위한 설계도 

컨테이너 : 실제 실행 중인 애플리케이션 환경

가상 환경을 위해 
	AWS EC2 인스턴스와 리눅스 서버를 사용하려면,

	- 클라우드와 EC2의 개념
    
	- 리눅스 기본 명령어와 환경
    
	- 네트워크와 보안(포트, 방화벽)
    
	- 소프트웨어 설치 및 서비스 실행
    
	- 도커 등 배포 도구

## 주요 설정 설명

- **db 서비스**
    
    - `image`: 사용할 DB 이미지(mysql:8.0 또는 postgres:15 등)
        
    - `environment`: DB 이름, 사용자, 비밀번호 등 환경 변수 지정
        
    - `volumes`: 데이터 영속성 보장(컨테이너 재시작/삭제에도 데이터 유지)
        
    - `healthcheck`: DB가 완전히 기동된 후 app 서비스가 시작되도록 보장
        
- **app 서비스**
    
    - `build`: 현재 디렉토리의 Dockerfile로 이미지 빌드
        
    - `command`: FastAPI 실행 명령(uvicorn 등)
        
    - `environment`: FastAPI에서 사용할 DATABASE_URL 환경 변수 지정
        
    - `depends_on`: db 서비스가 정상 기동된 후 app 실행
        
    - `ports`: 외부 접속용 포트 매핑(8000)
        
    - `volumes`: 코드 변경 시 핫리로드 등 개발 편의