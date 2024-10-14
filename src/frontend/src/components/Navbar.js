import { useState, useRef } from 'react';
import { Navbar as BootstrapNavbar, Nav, Container, Overlay } from 'react-bootstrap';
import '../styles/Navbar.css';

const Navbar = () => {
  const [showSubmenu, setShowSubmenu] = useState(false);
  const target = useRef(null);

  const handleToggle = () => setShowSubmenu(!showSubmenu);

  return (
    <>
      <BootstrapNavbar expand="lg" className="custom-navbar">
        <Container>
          <BootstrapNavbar.Brand href="#home"></BootstrapNavbar.Brand>
          <BootstrapNavbar.Toggle aria-controls="basic-navbar-nav" />
          <BootstrapNavbar.Collapse id="basic-navbar-nav">
            <Nav className="me-auto">
              <Nav.Link href="#home" style={{ marginRight: '20px' }}>Home</Nav.Link>
              <Nav.Link href="#link" style={{ marginRight: '20px' }}>Link</Nav.Link>
              <Nav.Link ref={target} onClick={handleToggle} style={{ marginRight: '20px' }}>
                Dropdown
              </Nav.Link>
            </Nav>
          </BootstrapNavbar.Collapse>
        </Container>
      </BootstrapNavbar>

      <Overlay target={target.current} show={showSubmenu} placement="bottom">
        {({ placement, arrowProps, show: _show, popper, ...props }) => (
          <div
            {...props}
            className="custom-overlay"
          >
            <Nav className="flex-row">
              <Nav.Link href="#action/3.1" style={{ marginRight: '20px' }}>Action</Nav.Link>
              <Nav.Link href="#action/3.2" style={{ marginRight: '20px' }}>Another action</Nav.Link>
              <Nav.Link href="#action/3.3" style={{ marginRight: '20px' }}>Something</Nav.Link>
              <Nav.Link href="#action/3.4" style={{ marginRight: '20px' }}>Separated link</Nav.Link>
            </Nav>
          </div>
        )}
      </Overlay>
    </>
  );
};

export default Navbar;