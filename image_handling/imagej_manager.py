import imagej
import scyjava


class ImageJManager:
    _instance = None  # Singleton instance

    def __new__(cls, version, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ImageJManager, cls).__new__(cls)
            cls._instance._initialized = False  # Flag to ensure one-time initialization
        return cls._instance

    def __init__(self, version):
        # Only initialize once
        if self._initialized:
            # Optional: Check if the new version matches the already set version
            if self.version != version:
                raise ValueError(f"ImageJManager is already initialized with version '{self.version}'.")
            return

        self.version = version
        self._initialize()  # Perform one-time initialization
        self._initialized = True

    def _initialize(self):
        """Initialize the ImageJ instance only once."""
        if not hasattr(self, "ij"):
            # Use the correct method to lower the version string
            if self.version.lower() == 'default':
                self.ij = imagej.init('sc.fiji:fiji', mode='interactive', add_legacy=True)
            else:
                self.ij = imagej.init(self.version, mode='interactive', add_legacy=True)

    def get_ij(self):
        """Provides access to the ImageJ instance."""
        return self.ij