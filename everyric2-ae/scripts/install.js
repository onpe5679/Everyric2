const fs = require('fs');
const path = require('path');
const os = require('os');

const EXTENSION_ID = 'com.everyric.ae';

function getExtensionsPath() {
  const platform = os.platform();
  const home = os.homedir();
  
  if (platform === 'darwin') {
    return path.join(home, 'Library/Application Support/Adobe/CEP/extensions');
  } else if (platform === 'win32') {
    return path.join(process.env.APPDATA || home, 'Adobe/CEP/extensions');
  } else {
    return path.join(home, '.config/Adobe/CEP/extensions');
  }
}

function copyRecursive(src, dest) {
  if (!fs.existsSync(src)) {
    console.error(`Source not found: ${src}`);
    return;
  }

  const stat = fs.statSync(src);
  
  if (stat.isDirectory()) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    
    const entries = fs.readdirSync(src);
    for (const entry of entries) {
      if (entry === 'node_modules' || entry === '.git' || entry === 'src') {
        continue;
      }
      copyRecursive(path.join(src, entry), path.join(dest, entry));
    }
  } else {
    fs.copyFileSync(src, dest);
  }
}

function enablePlayerDebugMode() {
  const platform = os.platform();
  const keyPath = platform === 'darwin' 
    ? '/Library/Preferences/com.adobe.CSXS.11.plist'
    : null;
  
  if (platform === 'win32') {
    console.log('\nTo enable debug mode on Windows:');
    console.log('1. Open Registry Editor (regedit)');
    console.log('2. Navigate to: HKEY_CURRENT_USER\\SOFTWARE\\Adobe\\CSXS.11');
    console.log('3. Create a new String value named "PlayerDebugMode" with value "1"');
    console.log('4. Restart After Effects');
  } else if (platform === 'darwin') {
    console.log('\nTo enable debug mode on macOS, run:');
    console.log('defaults write com.adobe.CSXS.11 PlayerDebugMode 1');
  }
}

function main() {
  const projectRoot = path.resolve(__dirname, '..');
  const extensionsPath = getExtensionsPath();
  const targetPath = path.join(extensionsPath, EXTENSION_ID);

  console.log('Installing Everyric2 After Effects Plugin...');
  console.log(`Source: ${projectRoot}`);
  console.log(`Target: ${targetPath}`);

  if (!fs.existsSync(path.join(projectRoot, 'dist', 'js', 'main.js'))) {
    console.error('\nError: Build files not found. Run "npm run build" first.');
    process.exit(1);
  }

  if (fs.existsSync(targetPath)) {
    console.log('Removing existing installation...');
    fs.rmSync(targetPath, { recursive: true, force: true });
  }

  if (!fs.existsSync(extensionsPath)) {
    fs.mkdirSync(extensionsPath, { recursive: true });
  }

  console.log('Copying files...');
  copyRecursive(projectRoot, targetPath);

  console.log('\nInstallation complete!');
  console.log(`Extension installed to: ${targetPath}`);
  
  enablePlayerDebugMode();
  
  console.log('\nTo use the plugin:');
  console.log('1. Restart After Effects');
  console.log('2. Go to Window > Extensions > Everyric2');
}

main();
