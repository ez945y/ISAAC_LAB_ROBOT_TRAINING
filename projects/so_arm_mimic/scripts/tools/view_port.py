def setup_camera_viewports(env_cfg, simulation_app):
    """
    Dynamically discovers cameras from the environment config, creates viewports,
    binds cameras to them, and arranges the windows.
    """
    try:
        import omni.kit.viewport.utility
        import omni.kit.commands
        import omni.usd
        from isaaclab.sensors import CameraCfg
        import re
        
        # 1. Discover Cameras from Config (Simple & Clean)
        camera_map = {}
        target_env_ns = "env_0"
        
        # Iterate attributes of the Scene Config
        for attr_name in dir(env_cfg.scene):
            if attr_name.startswith("__"): continue
            
            attr_val = getattr(env_cfg.scene, attr_name)
            if isinstance(attr_val, CameraCfg):
                # Found a camera config!
                raw_path = attr_val.prim_path
                
                # Fix path logic
                if "{ENV_REGEX_NS}" in raw_path:
                        actual_path = raw_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
                else:
                        # Handle regex path: /World/envs/env_.*/Robot/...
                        # Use non-greedy match for env segment (stop at first slash)
                        actual_path = re.sub(r"env_[^/]*", target_env_ns, raw_path)
                        
                camera_map[attr_name] = actual_path

        print(f"Dynamically discovered cameras from Config: {list(camera_map.keys())}")
        
        # 2. Create Viewports
        viewports = [] 
        
        stage = omni.usd.get_context().get_stage()

        for cam_name, cam_path in camera_map.items():
            # Validate
            prim = stage.GetPrimAtPath(cam_path)
            if not prim.IsValid():
                print(f"Error: Camera prim NOT found at: {cam_path}")
                # Attempt safe fallback using cam_path, NOT raw_path
                if "env_" in cam_path:
                        fallback = re.sub(r"env_[^/]*", "env_0", cam_path)
                        if stage.GetPrimAtPath(fallback).IsValid():
                            print(f"  Fixed path to: {fallback}")
                            cam_path = fallback
                        else:
                            print(f"  Could not auto-fix path. Skipping {cam_name}")
                            continue
                else:
                        continue

            title = f"Cam: {cam_name}"
            vp_win = omni.kit.viewport.utility.create_viewport_window(title)
            viewports.append((vp_win, cam_path))
        
        # Allow assets to spawn/render
        for _ in range(5): simulation_app.update()
        
        # 3. Bind Cameras (Robust)
        for vp_win, cam_path in viewports:
            if hasattr(vp_win, "viewport_api"):
                # Try using the Command system which handles UI updates better
                try:
                    omni.kit.commands.execute(
                        "SetViewportCamera",
                        viewport_api=vp_win.viewport_api,
                        camera_path=cam_path
                    )
                except Exception as e:
                    print(f"Cmd failed: {e}. Fallback to API.")
                    vp_win.viewport_api.set_active_camera(cam_path)
            else:
                print(f"Warning: Could not set camera for {cam_path} (No viewport_api)")

        # 4. Arrange Windows (Dynamic Tiling)
        def arrange_windows():
            width = 500
            height = 400
            start_x = 100
            start_y = 100
            
            for i, (win, _) in enumerate(viewports):
                if hasattr(win, "set_position"):
                    win.visible = True
                    win.width = width
                    win.height = height
                    win.position_x = start_x + (i * (width + 10))
                    win.position_y = start_y

        try:
            arrange_windows()
        except Exception as e:
            print(f"Warning: Auto-arrangement failed: {e}")

        print("Created additional viewports for cameras.")
        
    except ImportError:
        print("Warning: Could not import viewport utility. Skipping extra viewports.")
    except Exception as e:
        print(f"Warning: Failed to setup viewports: {e}")
