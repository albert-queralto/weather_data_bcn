import React, { useState } from "react";
import { Navbar, Nav, NavDropdown } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import "../styles/Navbar.css";

const HoverNavDropdown = ({ title, id, children }) => {
    const [show, setShow] = useState(false);

    const showDropdown = () => setShow(true);
    const hideDropdown = () => setShow(false);

    return (
        <NavDropdown
            title={title}
            id={id}
            show={show}
            onMouseEnter={showDropdown}
            onMouseLeave={hideDropdown}
        >
            {children}
        </NavDropdown>
    );
};

const renderMainMenu = (handleSubmenuSelection) => (
    <Navbar bg="dark" data-bs-theme="dark" expand="lg" className="navbar-custom">
        <Navbar.Brand href="#">
        </Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
            <Nav className="mr-auto">
                <HoverNavDropdown title="Management" id="gestion-dropdown">
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("usermanagement")}>Users</NavDropdown.Item>
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("preprocessors")}>Preprocessors</NavDropdown.Item>
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("models")}>Models</NavDropdown.Item>
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("outliers")}>Outliers</NavDropdown.Item>
                </HoverNavDropdown>
                <HoverNavDropdown title="Data Preprocessing" id="preprocessing-dropdown">
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("preprocessing-visualize")}>Visualize</NavDropdown.Item>
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("preprocessing-data")}>Preprocess data</NavDropdown.Item>
                </HoverNavDropdown>
                <HoverNavDropdown title="Training and predictions" id="training-dropdown">
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("training-training")}>Manual training</NavDropdown.Item>
                </HoverNavDropdown>
                <HoverNavDropdown title="Informes" id="reports-dropdown">
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("reports-predictions")}>Evolución predicciones</NavDropdown.Item>
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("reports-real-predictions")}>Evolución medidas reales y predicciones</NavDropdown.Item>
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("reports-descriptors-predictions")}>Evolución descriptores y predicciones</NavDropdown.Item>
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("reports-weights")}>Visualización de pesos</NavDropdown.Item>
                </HoverNavDropdown>
                <HoverNavDropdown title="Otros" id="other-dropdown">
                    <NavDropdown.Item href="#" onClick={() => handleSubmenuSelection("other-about")}>Acerca de</NavDropdown.Item>
                </HoverNavDropdown>
                <Nav.Link href="#" onClick={() => handleSubmenuSelection("logout")}>Logout</Nav.Link>
            </Nav>
        </Navbar.Collapse>
    </Navbar>
);

export default renderMainMenu;