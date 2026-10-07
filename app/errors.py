"""Errores con un mensaje simple para el usuario y detalle técnico para los logs."""


class UserFacingError(Exception):
    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.detail = detail or user_message


class InvalidURLError(UserFacingError):
    pass


class DownloadError(UserFacingError):
    pass


class TranscriptionError(UserFacingError):
    pass


class AnalysisError(UserFacingError):
    pass


class ClipGenerationError(UserFacingError):
    pass


class FFmpegMissingError(UserFacingError):
    def __init__(self, detail: str = ""):
        super().__init__(
            "No se encontró FFmpeg. Instalalo (ver README) y reiniciá LongToShort.", detail)
