# DCS için Lua + LuaRocks + Export.lua Kurulumu

## Gerekli Dosyalar

- [Lua ve LuaRocks klasörlerini buradan indir](https://drive.google.com/drive/folders/1v3Eu0eKolEBcNXIicZy2wTmtm1dTJhi0?usp=sharing)  

## Kurulum Adımları

1. **Lua klasörünü kopyalayın**  
   `C:\Users\{kullanıcınız}` dizinine taşıyın.

2. **LuaRocks klasörü**  
   İndirilenlerde kalabilir (`Downloads` dizininde).

3. **Ortam değişkenlerini ayarlayın**  
   - Windows arama çubuğuna **Sistem ortam değişkenlerini düzenle** yazın.  
   - **Ortam değişkenleri** → **Sistem değişkenleri** kısmında `Path` → **Düzenle** → **Yeni**  
   - Aşağıdaki pathleri ayrı ayrı ekleyin:  
     - `C:\Users\{kullanıcınız}\Lua`  
     - `C:\Users\{kullanıcınız}\Downloads\luarocks-3.13.0-windows-64`

4. **LuaRocks yapılandırması ve modül kurulumu**  
   CMD üzerinden LuaRocks dizinine gidin:  
   ```bash
   cd C:\Users\{kullanıcınız}\Downloads\luarocks-3.13.0-windows-64
   luarocks --local config --lua-version=5.1 variables.LUA "C:\Users\{kullanıcınız}\Lua\lua5.1.exe"
   luarocks install luasocket --lua-version=5.1

5. **Export.lua düzenlemesi**  
   `Export.lua` dosyasındaki paket yollarını kullanıcı adınıza göre düzenleyin:  
   ```lua
   package.path = package.path .. ";C:/Users/{kullanıcınız}/AppData/Roaming/luarocks/share/lua/5.1/?.lua"
   package.cpath = package.cpath .. ";C:/Users/{kullanıcınız}/AppData/Roaming/luarocks/lib/lua/5.1/socket/?.dll"

6. **Export.lua dosyasını Scripts klasörüne kopyalayın**  
   - Düzenlediğiniz `Export.lua` dosyasını aşağıdaki dizine kopyalayın:  
     ```
     C:\Users\{kullanıcınız}\Saved Games\DCS\Scripts
     ```
   - Eğer `Scripts` klasörü yoksa kendiniz oluşturun.  

7. **Listener'ı çalıştırın**

- dcs_listener.py'yi anaconda prompt benzeri python'ın çalışabileceği bir ortamda çalıştırın.

8. **Simülasyona girip bir mission başlatın** 

9. **dcs_listener.py dosyasını çalıştırdığınız terminalden simülasyon verilerinin aktığını görün**