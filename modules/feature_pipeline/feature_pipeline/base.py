from threading import Lock


class SingletonMeta(type):
    """C'est une implémentation d'un fil sûr d'un Singleton."""
    
    _instances = {}
    
    _lock = Lock()
    
    """
    Nous avons maintenant un objet lock qui va être utilisé pour synchroniser les threads durant
    le premier accès au Signleton.
    """
    
    def __call__(cls, *args, **kwargs):       
        """
        Les changements possibles de la valeur de l'argument '__init__'  n'affecteront pas l'instance renvoyée.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args,**kwargs)
                cls._instances[cls] = instance
                
        return cls._instances[cls]